from markdown_pdf import MarkdownPdf, Section


def save_markdown_response(file_path: str, query: str, response: str) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Query: {query}\n\n{response}")


def save_pdf_response(file_path: str, query: str, response: str) -> None:
    pdf = MarkdownPdf()
    pdf.add_section(Section(f"# Query: {query}\n\n{response}"))
    pdf.save(file_path)
