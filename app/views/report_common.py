"""
Funcionalidade compartilhada entre as janelas de relatório de explicabilidade
(Árvore, Regressão Logística, KNN, Random Forest, SVM e Comitê).
"""

import tkinter
from tkinter import filedialog, messagebox

from core.decision import ROTULO_MALIGNO, ROTULO_REVISAR
from core.fatores import linhas_fatores
from utils.pdf_report import export_patient_report, resolve_reports_dir

COR_MALIGNO = "#e74c3c"
COR_BENIGNO = "#2ecc71"
COR_REVISAR = "#e67e22"


def cor_da_classe(classe: str) -> str:
    """
    Cor de um paciente segundo o que o modelo decidiu sobre ele.

    Existe porque a decisão deixou de ser binária: com a recusa ligada, pintar
    "Revisar" com a cor de Benigno (o comportamento de um ``if maligno else``)
    faria um caso não decidido parecer um caso liberado — a confusão mais cara
    possível nesta tela.
    """
    if classe == ROTULO_MALIGNO:
        return COR_MALIGNO
    return COR_REVISAR if classe == ROTULO_REVISAR else COR_BENIGNO


class PatientPDFExportMixin:
    """
    Adiciona a uma janela de relatório a capacidade de exportar, em PDF, a
    explicação do paciente atualmente selecionado.

    Pressupõe que a subclasse (um ``ctk.CTkToplevel``) mantenha um
    ``ttk.Treeview`` em ``self._tree`` (lista de pacientes, selecionável) e
    um ``ctk.CTkTextbox`` em ``self._detalhe`` (texto explicativo do
    paciente selecionado). Se a janela também mantiver um gráfico
    matplotlib em ``self._canvas`` (mapa UMAP, vizinhos, etc.), a figura é
    embutida no PDF. Uma janela com mais de um gráfico (caso do SVM: mapa +
    margem) pode expor ``_figuras_pdf()`` e devolver todos eles.
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

        # O bloco de fatores pode ainda estar sendo calculado: o PDF não pode
        # sair com "calculando…" no lugar dele.
        if hasattr(self, "_garantir_fatores"):
            self._garantir_fatores()

        try:
            if hasattr(self, "_figuras_pdf"):
                figura = self._figuras_pdf()
            else:
                figura = self._canvas.figure if hasattr(self, "_canvas") else None
            export_patient_report(
                caminho, self.title(), paciente,
                self._detalhe.get("1.0", "end"), figura=figura,
            )
            messagebox.showinfo("Exportado", f"Relatório salvo em:\n{caminho}", parent=self)
        except Exception as e:
            messagebox.showerror("Erro ao exportar", str(e), parent=self)


class FatoresPacienteMixin:
    """
    Põe, no detalhe de cada paciente, o bloco "Fatores que mais pesaram".

    É o que deixa a explicação por fator igual em todas as janelas por modelo:
    o bloco entra logo abaixo do diagnóstico (depois do primeiro parágrafo do
    texto de ``_formatar_detalhe``), no mesmo formato para todos.

    O cálculo leva de frações de segundo a ~6 s (comitê), então não pode travar
    a janela: o detalhe aparece na hora, com uma linha "calculando…", e o bloco
    entra quando fica pronto — se o mesmo paciente ainda estiver selecionado.
    O trabalho roda numa thread do ``FatoresDoLote``; esta classe só consulta o
    resultado a partir do laço do Tk, que nunca é tocado de outra thread.

    Pressupõe ``self._detalhe`` (CTkTextbox) e ``self._formatar_detalhe(e)``.
    A subclasse declara ``MODELO_FATORES``, chama ``_configurar_fatores`` no
    construtor e ``_escrever_detalhe(e)`` ao selecionar um paciente.
    """

    MODELO_FATORES = None
    _INTERVALO_MS = 200

    def _configurar_fatores(self, fatores):
        """
        Guarda o provedor de fatores do lote (``core.fatores.FatoresDoLote``).

        Parameters
        ----------
        fatores : FatoresDoLote ou None
            None quando não há como calcular (ex.: sessão antiga do histórico
            sem o lote salvo) — o detalhe sai sem o bloco, como antes.
        """
        self._fatores = fatores
        self._paciente_em_exibicao = None

    def _fatores_ativos(self) -> bool:
        """True se esta janela tem como calcular os fatores do seu modelo."""
        fatores = getattr(self, "_fatores", None)
        return (fatores is not None and self.MODELO_FATORES is not None
                and fatores.disponivel(self.MODELO_FATORES))

    def _texto_com_fatores(self, e: dict, bloco: str) -> str:
        """Insere o bloco logo após o primeiro parágrafo (o do diagnóstico)."""
        texto = self._formatar_detalhe(e)
        if not bloco:
            return texto
        cabeca, separador, resto = texto.partition("\n\n")
        if not separador:
            return f"{texto}\n\n{bloco}"
        return f"{cabeca}\n\n{bloco}\n\n{resto}"

    def _bloco(self, e: dict, resultado=None, erro=None) -> str:
        """Texto do bloco no estado atual: pronto, com erro ou calculando."""
        if resultado is not None:
            modelo, rotulo = self.MODELO_FATORES, e.get('classe')
            return "\n".join(linhas_fatores(
                resultado, modelo, rotulo,
                regra=self._fatores.regra(modelo),
                corte=self._fatores.corte(modelo, rotulo)))
        if erro is not None:
            return f"Fatores que mais pesaram: não foi possível calcular ({erro})."
        return "Fatores que mais pesaram: calculando… (alguns segundos)"

    def _substituir_detalhe(self, texto: str):
        """Troca o conteúdo da caixa de detalhe (que fica somente leitura)."""
        self._detalhe.configure(state="normal")
        self._detalhe.delete("1.0", "end")
        self._detalhe.insert("1.0", texto)
        self._detalhe.configure(state="disabled")

    def _escrever_detalhe(self, e: dict):
        """
        Mostra o detalhe do paciente e, se preciso, agenda o cálculo dos fatores.

        Parameters
        ----------
        e : dict
            Explicação do paciente selecionado (precisa de 'indice').
        """
        self._paciente_em_exibicao = e
        if not self._fatores_ativos():
            self._substituir_detalhe(self._formatar_detalhe(e))
            return

        futuro = self._fatores.agendar(self.MODELO_FATORES, e['indice'])
        if futuro.done():
            self._renderizar_quando_pronto(e, futuro)
            return
        self._substituir_detalhe(self._texto_com_fatores(e, self._bloco(e)))
        self.after(self._INTERVALO_MS, lambda: self._acompanhar(e, futuro))

    def _acompanhar(self, e: dict, futuro):
        """Consulta o cálculo pelo laço do Tk até ele terminar."""
        try:
            if not self.winfo_exists():
                return
        except tkinter.TclError:
            return    # a janela foi fechada com o cálculo em andamento
        if not futuro.done():
            self.after(self._INTERVALO_MS, lambda: self._acompanhar(e, futuro))
            return
        if self._paciente_em_exibicao is e:
            self._renderizar_quando_pronto(e, futuro)

    def _renderizar_quando_pronto(self, e: dict, futuro):
        """Redesenha o detalhe com o bloco final (ou com o erro do cálculo)."""
        try:
            bloco = self._bloco(e, resultado=futuro.result())
        except Exception as erro:
            bloco = self._bloco(e, erro=erro)
        self._substituir_detalhe(self._texto_com_fatores(e, bloco))

    def _garantir_fatores(self):
        """
        Espera o cálculo do paciente exibido terminar e atualiza o detalhe.

        Chamado antes de exportar o PDF, que copia o texto da caixa de detalhe.
        Bloqueia a janela por alguns segundos no pior caso — aceitável numa
        ação explícita de exportação, e melhor do que um PDF incompleto.
        """
        e = getattr(self, "_paciente_em_exibicao", None)
        if e is None or not self._fatores_ativos():
            return
        futuro = self._fatores.agendar(self.MODELO_FATORES, e['indice'])
        try:
            futuro.result(timeout=120)
        except Exception:
            pass      # o erro aparece no próprio bloco, via _renderizar_quando_pronto
        if futuro.done():
            self._renderizar_quando_pronto(e, futuro)
