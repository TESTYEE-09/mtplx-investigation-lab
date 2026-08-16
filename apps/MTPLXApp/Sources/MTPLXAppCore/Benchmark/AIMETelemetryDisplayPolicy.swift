import Foundation

public enum AIMETelemetryDisplayPolicy {
    public static let liveDecodeWarmupCompletionTokens = 0

    public static func displayedDecodeTokS(
        candidate: Double?,
        source: String,
        completionTokens: Int?,
        minimumCompletionTokens: Int = liveDecodeWarmupCompletionTokens
    ) -> Double? {
        guard let candidate, candidate.isFinite, candidate > 0 else {
            return nil
        }
        if source == "inflight_exact",
           minimumCompletionTokens > 0,
           (completionTokens ?? 0) < minimumCompletionTokens {
            return nil
        }
        return candidate
    }

    public static func suppressesLiveDecode(
        candidate: Double?,
        source: String,
        completionTokens: Int?,
        minimumCompletionTokens: Int = liveDecodeWarmupCompletionTokens
    ) -> Bool {
        candidate != nil
            && displayedDecodeTokS(
                candidate: candidate,
                source: source,
                completionTokens: completionTokens,
                minimumCompletionTokens: minimumCompletionTokens
            ) == nil
    }
}
