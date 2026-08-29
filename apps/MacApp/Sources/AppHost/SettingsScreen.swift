import AppCore
import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                DetailLevelCard()
                if model.detail != .simple { DecodingCard() }
                if let report = model.doctorReport { MachineCard(report: report) }
                ServerCard()
                ModelFoldersCard()
                AboutCard()
            }
            .padding(16)
        }
        .task { await refreshDiagnostics() }
    }

    private func refreshDiagnostics() async {
        await model.refreshDiagnostics()
    }
}
