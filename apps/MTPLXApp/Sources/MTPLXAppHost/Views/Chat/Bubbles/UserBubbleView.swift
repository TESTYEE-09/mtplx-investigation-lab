import SwiftUI
import MTPLXAppCore

// MARK: - UserBubbleView
//
// Right-anchored bubble for `role == .user` turns. Brand.cardSurface
// fill + a chrome hairline stroke + asymmetric corners (large 14pt on
// three corners, small 4pt on bottom-trailing — the tail side).
// Attachments render above the bubble via `MessageAttachmentStrip`.
// Max bubble width 576pt keeps long pastes readable.
//
// Jet Chrome pass: the V0 cool-blue tint is gone. Right-alignment and
// the asymmetric corner already differentiate user from assistant
// visually — adding a color tint was redundant ornament that read as
// "AI app blue chat bubble." Neutral cardSurface + hairlineStrong
// chrome border reads as polished, not as branded blue.

struct UserBubbleView: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 60)
            VStack(alignment: .trailing, spacing: 6) {
                if !message.attachments.isEmpty {
                    MessageAttachmentStrip(attachments: message.attachments)
                }
                if !message.visibleContent.isEmpty {
                    Text(message.visibleContent)
                        .font(.system(size: 14))
                        .foregroundStyle(Brand.typeHi)
                        .multilineTextAlignment(.leading)
                        .frame(alignment: .leading)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 11)
                        .background(
                            UnevenRoundedRectangle(
                                topLeadingRadius: 14,
                                bottomLeadingRadius: 14,
                                bottomTrailingRadius: 4,
                                topTrailingRadius: 14,
                                style: .continuous
                            )
                            .fill(Brand.cardSurface)
                            .overlay(
                                UnevenRoundedRectangle(
                                    topLeadingRadius: 14,
                                    bottomLeadingRadius: 14,
                                    bottomTrailingRadius: 4,
                                    topTrailingRadius: 14,
                                    style: .continuous
                                )
                                .stroke(Brand.separatorStrong, lineWidth: Brand.hairlineStrong)
                            )
                        )
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: 576, alignment: .trailing)
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }
}

// MARK: - MessageAttachmentStrip

struct MessageAttachmentStrip: View {
    let attachments: [ChatAttachment]

    var body: some View {
        if attachments.count == 1, let attachment = attachments.first {
            AttachmentCard(
                filename: attachment.filename,
                fileExtension: Self.fileExtension(filename: attachment.filename),
                sizeBytes: attachment.sizeBytes,
                imageData: attachment.imageData
            )
        } else {
            FlowRow(horizontalSpacing: 8, verticalSpacing: 8, alignment: .trailing) {
                ForEach(attachments, id: \.id) { attachment in
                    AttachmentCard(
                        filename: attachment.filename,
                        fileExtension: Self.fileExtension(filename: attachment.filename),
                        sizeBytes: attachment.sizeBytes,
                        imageData: attachment.imageData
                    )
                }
            }
            .frame(maxWidth: 576)
        }
    }

    private static func fileExtension(filename: String) -> String {
        guard let dot = filename.lastIndex(of: ".") else { return "" }
        return String(filename[filename.index(after: dot)...])
    }
}
