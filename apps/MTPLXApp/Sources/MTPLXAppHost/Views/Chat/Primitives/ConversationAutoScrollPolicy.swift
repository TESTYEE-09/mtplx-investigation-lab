import Foundation

// MARK: - ConversationAutoScrollPolicy
//
// Port of Aphanes V2's `ConversationAutoScrollPolicy`. Decides whether
// the conversation view should auto-scroll to the bottom as new
// streamed tokens arrive. The two thresholds (detach at >120pt, reattach
// at <28pt) come from Aphanes' production tuning.
//
// Usage:
//   1. ChatConversationView observes the scroll view's distance from
//      the bottom and calls `didScroll(distanceToBottom:)` per change.
//   2. When a streaming update arrives the view consults
//      `shouldAutoScrollForStreamingUpdate`. If true, scroll to bottom.
//   3. When the user sends a new message the view force-scrolls and
//      calls `attach()` to rearm the policy.
//
// Pure value type so it can be held inline on the view and mutated
// without spawning a class.

public enum ConversationAutoScrollAction: Sendable, Equatable {
    case immediate
    case deferred
}

public struct ConversationAutoScrollPolicy: Sendable, Equatable {
    public static let detachThreshold: CGFloat = 120
    public static let reattachThreshold: CGFloat = 28

    public private(set) var isPinnedToBottom: Bool
    public private(set) var isStreaming: Bool

    public init(isPinnedToBottom: Bool = true, isStreaming: Bool = false) {
        self.isPinnedToBottom = isPinnedToBottom
        self.isStreaming = isStreaming
    }

    public mutating func didAppear() -> [ConversationAutoScrollAction] {
        isPinnedToBottom = true
        isStreaming = false
        return [.immediate]
    }

    public mutating func didOpenConversation() -> [ConversationAutoScrollAction] {
        isPinnedToBottom = true
        isStreaming = false
        return [.immediate]
    }

    public mutating func didSendUserMessage() -> [ConversationAutoScrollAction] {
        isPinnedToBottom = true
        return [.immediate]
    }

    public mutating func didStartStreaming() -> [ConversationAutoScrollAction] {
        isStreaming = true
        isPinnedToBottom = true
        return [.immediate]
    }

    public mutating func didFinishStreaming() -> [ConversationAutoScrollAction] {
        isStreaming = false
        return isPinnedToBottom ? [.immediate, .deferred] : []
    }

    /// Whether the scroll container should follow a fresh streaming
    /// token by scrolling to the bottom.
    public var shouldAutoScrollForStreamingUpdate: Bool {
        isPinnedToBottom
    }

    /// Reset to "follow streaming". Called when the user submits a new
    /// turn or explicitly taps "scroll to bottom".
    public mutating func attach() {
        isPinnedToBottom = true
    }

    /// Mark the policy as detached. Called when the user manually
    /// scrolls away from the bottom.
    public mutating func detach() {
        isPinnedToBottom = false
    }

    /// React to a scroll event. Detaches when the user scrolls > 120pt
    /// off the bottom, reattaches when they come back within 28pt.
    /// Returns true when the pinned state changed so the host view can
    /// trigger any side-effects (toast, scroll-to-bottom pill, etc.).
    @discardableResult
    public mutating func didScroll(
        distanceToBottom: CGFloat,
        isUserInitiated: Bool = true
    ) -> [ConversationAutoScrollAction] {
        if distanceToBottom <= Self.reattachThreshold {
            isPinnedToBottom = true
            return []
        }

        if isUserInitiated, distanceToBottom >= Self.detachThreshold {
            isPinnedToBottom = false
        }

        return []
    }
}
