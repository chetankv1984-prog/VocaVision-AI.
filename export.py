from fpdf import FPDF
def to_pdf(title, text):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text.encode("latin-1", "replace").decode("latin-1"))
    return bytes(pdf.output())
