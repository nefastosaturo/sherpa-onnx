// sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.h"

#include <algorithm>
#include <deque>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

// Helper structure to track hypothesis with decoder state
struct NeMoHypothesis {
  std::vector<int32_t> ys;           // token sequence
  std::vector<int32_t> timestamps;   // timestamps for each token
  float log_prob;                     // accumulated log probability
  std::vector<Ort::Value> decoder_states;  // RNN/LSTM states
  const ContextState *context_state;  // context graph state
  
  NeMoHypothesis() : log_prob(0.0f), context_state(nullptr) {}
  
  // Copy constructor - needed for hypothesis expansion
  NeMoHypothesis(const NeMoHypothesis &other)
      : ys(other.ys),
        timestamps(other.timestamps),
        log_prob(other.log_prob),
        context_state(other.context_state) {
    // Deep copy of decoder states
    decoder_states.reserve(other.decoder_states.size());
    for (const auto &state : other.decoder_states) {
      decoder_states.push_back(Clone(state));
    }
  }
  
  NeMoHypothesis &operator=(const NeMoHypothesis &other) {
    if (this != &other) {
      ys = other.ys;
      timestamps = other.timestamps;
      log_prob = other.log_prob;
      context_state = other.context_state;
      
      decoder_states.clear();
      decoder_states.reserve(other.decoder_states.size());
      for (const auto &state : other.decoder_states) {
        decoder_states.push_back(Clone(state));
      }
    }
    return *this;
  }
  
  NeMoHypothesis(NeMoHypothesis &&) = default;
  NeMoHypothesis &operator=(NeMoHypothesis &&) = default;
};

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerModifiedBeamSearchNeMoDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    OfflineStream **ss /*= nullptr*/, int32_t n /*= 0*/) {
  
  PackedSequence packed_encoder_out = PackPaddedSequence(
      model_->Allocator(), &encoder_out, &encoder_out_length);

  int32_t batch_size =
      static_cast<int32_t>(packed_encoder_out.sorted_indexes.size());

  if (ss != nullptr) SHERPA_ONNX_CHECK_EQ(batch_size, n);

  int32_t vocab_size = model_->VocabSize();
  
  std::deque<std::vector<NeMoHypothesis>> finalized;
  std::vector<std::vector<NeMoHypothesis>> cur(batch_size);
  
  std::vector<ContextGraphPtr> context_graphs(batch_size, nullptr);
  
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Initialize: create initial hypothesis for each utterance
  for (int32_t i = 0; i < batch_size; ++i) {
    const ContextState *context_state = nullptr;
    if (ss != nullptr) {
      context_graphs[i] =
          ss[packed_encoder_out.sorted_indexes[i]]->GetContextGraph();
      if (context_graphs[i] != nullptr) {
        context_state = context_graphs[i]->Root();
      }
    }
    
    NeMoHypothesis blank_hyp;
    blank_hyp.ys = {0};  // Start with blank token
    blank_hyp.log_prob = 0.0f;
    blank_hyp.context_state = context_state;
    blank_hyp.decoder_states = model_->GetDecoderInitStates(1);
    
    cur[i].push_back(std::move(blank_hyp));
  }

  int32_t start = 0;
  int32_t t = 0;
  
  // Process encoder output frame by frame
  for (auto n : packed_encoder_out.batch_sizes) {
    Ort::Value cur_encoder_out = packed_encoder_out.Get(start, n);
    start += n;

    // Finalize utterances that are done
    if (n < static_cast<int32_t>(cur.size())) {
      for (int32_t k = static_cast<int32_t>(cur.size()) - 1; k >= n; --k) {
        finalized.push_front(std::move(cur[k]));
      }
      cur.erase(cur.begin() + n, cur.end());
    }

    // Expand each hypothesis
    std::vector<std::vector<NeMoHypothesis>> next_hyps(n);
    
    for (int32_t i = 0; i < n; ++i) {
      std::vector<std::pair<float, NeMoHypothesis>> all_candidates;
      
      // Get encoder output for this utterance
      Ort::Value encoder_out_i = Slice(model_->Allocator(), &cur_encoder_out, 0, i, i + 1);
      // Shape: (1, encoder_dim)
      
      // Process each hypothesis
      for (auto &hyp : cur[i]) {
        // Prepare decoder input: last token
        int32_t last_token = hyp.ys.back();
        std::array<int64_t, 2> decoder_input_shape = {1, 1};
        std::vector<int32_t> decoder_input_data = {last_token};
        
        Ort::Value decoder_input = Ort::Value::CreateTensor(
            memory_info, decoder_input_data.data(), 1,
            decoder_input_shape.data(), decoder_input_shape.size());
        
        std::array<int64_t, 1> decoder_input_length_shape = {1};
        std::vector<int32_t> decoder_input_length_data = {1};
        
        Ort::Value decoder_input_length = Ort::Value::CreateTensor(
            memory_info, decoder_input_length_data.data(), 1,
            decoder_input_length_shape.data(), decoder_input_length_shape.size());
        
        // Run decoder with current states
        auto decoder_result = model_->RunDecoder(
            std::move(decoder_input),
            std::move(decoder_input_length),
            std::move(hyp.decoder_states));
        
        Ort::Value decoder_out = std::move(decoder_result.first);
        std::vector<Ort::Value> next_states = std::move(decoder_result.second);
        
        // Run joiner
        Ort::Value logit = model_->RunJoiner(
            Clone(encoder_out_i),
            View(&decoder_out));
        
        float *p_logit = logit.GetTensorMutableData<float>();
        
        // Apply blank penalty
        if (blank_penalty_ > 0.0f) {
          p_logit[0] -= blank_penalty_;  // blank is at index 0
        }
        
        // Compute log softmax
        LogSoftmax(p_logit, vocab_size, 1);
        
        // Create candidates for all possible tokens
        for (int32_t token = 0; token < vocab_size; ++token) {
          NeMoHypothesis new_hyp;
          new_hyp.ys = hyp.ys;
          new_hyp.timestamps = hyp.timestamps;
          new_hyp.context_state = hyp.context_state;
          new_hyp.log_prob = hyp.log_prob + p_logit[token];
          
          // Deep copy decoder states for this hypothesis
          new_hyp.decoder_states.reserve(next_states.size());
          for (const auto &state : next_states) {
            new_hyp.decoder_states.push_back(Clone(state));
          }
          
          float context_score = 0.0f;
          
          // If not blank and not unk, add to sequence
          if (token != 0 && token != unk_id_) {
            new_hyp.ys.push_back(token);
            new_hyp.timestamps.push_back(t);
            
            // Update context graph
            if (context_graphs[i] != nullptr) {
              auto context_res = context_graphs[i]->ForwardOneStep(
                  new_hyp.context_state, token, false);
              context_score = std::get<0>(context_res);
              new_hyp.context_state = std::get<1>(context_res);
            }
            new_hyp.log_prob += context_score;
          }
          
          all_candidates.emplace_back(new_hyp.log_prob, std::move(new_hyp));
        }
      }
      
      // Keep top-k hypotheses
      std::partial_sort(
          all_candidates.begin(),
          all_candidates.begin() + std::min(max_active_paths_, 
                                           static_cast<int32_t>(all_candidates.size())),
          all_candidates.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });
      
      int32_t keep = std::min(max_active_paths_, 
                             static_cast<int32_t>(all_candidates.size()));
      next_hyps[i].reserve(keep);
      for (int32_t k = 0; k < keep; ++k) {
        next_hyps[i].push_back(std::move(all_candidates[k].second));
      }
    }
    
    cur = std::move(next_hyps);
    ++t;
  }

  // Add finalized utterances back
  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  // Finalize context biasing
  for (int32_t i = 0; i < cur.size(); ++i) {
    for (auto &hyp : cur[i]) {
      if (context_graphs[i] != nullptr) {
        auto context_res = context_graphs[i]->Finalize(hyp.context_state);
        hyp.log_prob += context_res.first;
        hyp.context_state = context_res.second;
      }
    }
  }

  // LM rescoring if available
  if (lm_) {
    // Convert to standard Hypotheses format for LM scoring
    std::vector<Hypotheses> lm_hyps(batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
      for (const auto &nemo_hyp : cur[i]) {
        Hypothesis h;
        h.ys = nemo_hyp.ys;
        h.log_prob = nemo_hyp.log_prob;
        h.timestamps = nemo_hyp.timestamps;
        h.context_state = nemo_hyp.context_state;
        lm_hyps[i].Add(std::move(h));
      }
    }
    
    lm_->ComputeLMScore(lm_scale_, 1, &lm_hyps);  // context_size=1 for NeMo
    
    // Copy LM scores back
    for (int32_t i = 0; i < batch_size; ++i) {
      auto most_probable = lm_hyps[i].GetMostProbable(true);
      // Find matching hypothesis and update
      for (auto &nemo_hyp : cur[i]) {
        if (nemo_hyp.ys == most_probable.ys) {
          nemo_hyp.log_prob = most_probable.log_prob;
          break;
        }
      }
    }
  }

  // Extract results
  std::vector<OfflineTransducerDecoderResult> unsorted_ans(batch_size);
  for (int32_t i = 0; i < batch_size; ++i) {
    // Find best hypothesis
    auto best_it = std::max_element(
        cur[i].begin(), cur[i].end(),
        [](const NeMoHypothesis &a, const NeMoHypothesis &b) {
          return a.log_prob < b.log_prob;
        });
    
    auto &r = unsorted_ans[packed_encoder_out.sorted_indexes[i]];
    
    // Strip leading blank token
    r.tokens = {best_it->ys.begin() + 1, best_it->ys.end()};
    r.timestamps = best_it->timestamps;
    
    // TDT mode: extract durations if needed
    if (is_tdt_) {
      // TDT models may need special handling for durations
      // This depends on your specific TDT implementation
      // For now, we just copy timestamps as-is
    }
  }

  return unsorted_ans;
}

}  // namespace sherpa_onnx
