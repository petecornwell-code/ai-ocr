def run_ocr_analysis(extracted_text: str) -> str:
    from crewai import Agent, Crew, Task

    analyst = Agent(
        role="Document Analyst",
        goal="Analyze OCR-extracted text and provide structured insights",
        backstory=(
            "You are an expert document analyst skilled at interpreting "
            "raw OCR output. You clean up text, identify document structure, "
            "and extract key information."
        ),
        verbose=False,
    )

    analysis_task = Task(
        description=(
            f"Analyze the following OCR-extracted text. Identify the document type, "
            f"extract key fields, correct obvious OCR errors, and provide a structured "
            f"summary.\n\nExtracted text:\n{extracted_text}"
        ),
        expected_output=(
            "A structured analysis containing: document type, key fields extracted, "
            "corrected text, and a brief summary."
        ),
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)
