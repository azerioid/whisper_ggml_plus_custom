// ignore_for_file: invalid_annotation_target

import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:whisper_ggml_plus/src/models/responses/whisper_transcribe_segment.dart';

part 'whisper_transcribe_response.freezed.dart';
part 'whisper_transcribe_response.g.dart';

/// Response model of whisper getVersion
@freezed
abstract class WhisperTranscribeResponse with _$WhisperTranscribeResponse {
  ///
  const factory WhisperTranscribeResponse({
    @JsonKey(name: '@type') required String type,
    required String text,
    @JsonKey(name: 'segments')
    required List<WhisperTranscribeSegment>? segments,

    /// ISO 639-1 code whisper used; the detected one when the request's
    /// language was "auto". Null with older native builds.
    String? language,
  }) = _WhisperTranscribeResponse;

  const WhisperTranscribeResponse._();

  /// Parse [json] to WhisperTranscribeResponse
  factory WhisperTranscribeResponse.fromJson(Map<String, dynamic> json) =>
      _$WhisperTranscribeResponseFromJson(json);
}
