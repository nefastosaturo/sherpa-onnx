// sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-lm.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"

namespace sherpa_onnx {

class OfflineTransducerModifiedBeamSearchNeMoDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerModifiedBeamSearchNeMoDecoder(
      OfflineTransducerNeMoModel *model,
      OfflineLM *lm,
      int32_t max_active_paths,
      float lm_scale,
      int32_t unk_id,
      float blank_penalty,
      bool is_tdt)
      : model_(model),
        lm_(lm),
        max_active_paths_(max_active_paths),
        lm_scale_(lm_scale),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty),
        is_tdt_(is_tdt) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out,
      Ort::Value encoder_out_length,
      OfflineStream **ss = nullptr,
      int32_t n = 0) override;

 private:
  OfflineTransducerNeMoModel *model_;  // Not owned
  OfflineLM *lm_;                      // Not owned; may be nullptr

  int32_t max_active_paths_;
  float lm_scale_;  // used only when lm_ is not nullptr
  int32_t unk_id_;
  float blank_penalty_;
  bool is_tdt_;  // Token-and-Duration Transducer mode
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
