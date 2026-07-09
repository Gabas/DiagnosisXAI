"""
Funcionalidade compartilhada entre as janelas de relatório de explicabilidade
(Árvore, Regressão Logística, KNN, Random Forest e SVM).
"""

from tkinter import filedialog, messagebox

from utils.pdf_report import export_patient_report, resolve_reports_dir


class PatientPDFExportMixin:
    """
    Adiciona a uma janela de relatório a capacidade de exportar, em PDF, a
    explicação do paciente atualmente selecionado.

    Pressupõe que a subclasse (um ``ctk.CTkToplevel``) mantenha um
    ``ttk.Treeview`` em ``self._tree`` (lista de pacientes, selecionável) e
    um ``ctk.CTkTextbox`` em ``self._detalhe`` (texto explicativo do
    paciente selecionado). Se a janela também mantiver um gráfico
    matplotlib em ``self._canvas`` (mapa UMAP, vizinhos, etc.), a figura é
    embutida no PDF.
    """

    def _exportar_pdf_paciente(self):
        """Pede o destino do arquivo e grava o PDF do paciente selecionado."""
        selecao = self._tree.selection()
        paciente = selecao[0] if selecao else "lote"
        sufixo = self.title().split("—")[-1].strip().lower().replace(" ", "_") or "relatorio"

        caminho = filedialog.asksaveasfilename(
            title="Salvar relatório do paciente",
            initialdir=resolve_reports_dir(),
            initialfile=f"paciente_{paciente}_{sufixo}.pdf",
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"), ("Todos os arquivos", "*.*")),
            parent=self,
        )
        if not caminho:
            return

        try:
            figura = self._canvas.figure if hasattr(self, "_canvas") else None
            export_patient_report(
                caminho, self.title(), paciente,
                self._detalhe.get("1.0", "end"), figura=figura,
            )
            messagebox.showinfo("Exportado", f"Relatório salvo em:\n{caminho}", parent=self)
        except Exception as e:
            messagebox.showerror("Erro ao exportar", str(e), parent=self)
