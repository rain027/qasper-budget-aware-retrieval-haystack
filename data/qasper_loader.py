# data/qasper_loader.py

import json
from typing import List, Dict, Optional
from haystack import Document
from tqdm import tqdm


class QASPERLoader:
    def __init__(self, qasper_path: str):
        with open(qasper_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_paper_ids(self) -> List[str]:
        return list(self.data.keys())

    def get_documents_for_indexing(
        self, paper_ids: Optional[List[str]] = None
    ) -> List[Document]:

        if paper_ids is None:
            paper_ids = self.get_paper_ids()

        documents = []

        for pid in tqdm(paper_ids, desc="Creating Documents"):
            paper = self.data[pid]
            sections = self._get_sections(paper)

            for s_idx, section in enumerate(sections):
                paragraphs = [
                    p.strip()
                    for p in section["content"].split("\n\n")
                    if len(p.strip()) > 50
                ]

                for p_idx, para in enumerate(paragraphs):
                    documents.append(
                        Document(
                            content=para,
                            meta={
                                "paper_id": pid,
                                "section_name": section["title"],
                                "section_type": section["section_type"],
                                "section_idx": s_idx,
                                "paragraph_idx": p_idx,
                                "position": s_idx / max(len(sections), 1),
                                "word_count": len(para.split()),
                            },
                        )
                    )

        return documents

    def _get_sections(self, paper: Dict) -> List[Dict]:
        sections = [
            {
                "title": "Abstract",
                "content": str(paper.get("abstract", "")),
                "section_type": "abstract",
            }
        ]

        full_text = paper.get("full_text", {})

        for name, paras in full_text.items():
            content_chunks = []

            # paras can be: list[str], list[list[str]], or str
            if isinstance(paras, list):
                for p in paras:
                    if isinstance(p, list):
                        # flatten nested lists
                        content_chunks.extend(str(x) for x in p if x)
                    elif isinstance(p, str):
                        content_chunks.append(p)
            elif isinstance(paras, str):
                content_chunks.append(paras)

            content = "\n\n".join(content_chunks)

            sections.append(
                {
                    "title": name,
                    "content": content,
                    "section_type": self._classify(name),
                }
            )

        return sections


    def _classify(self, name: str) -> str:
        n = name.lower()
        if "method" in n:
            return "methodology"
        if "result" in n or "experiment" in n:
            return "results"
        if "conclusion" in n:
            return "conclusion"
        if "intro" in n:
            return "introduction"
        return "body"

    def get_questions(self, paper_ids):
        questions = []

        for pid in paper_ids:
            paper = self.data[pid]
            qas = paper.get("qas", {})

            question_texts = qas.get("question", [])
            answers_list = qas.get("answers", [])
            question_ids = qas.get("question_id", [])

            for i, qtext in enumerate(question_texts):
                qid = question_ids[i] if i < len(question_ids) else f"q{i}"
                answers = answers_list[i] if i < len(answers_list) else None

                evidence = []
                extractive_spans = []

                if answers:
                    # Case 1: answers is a list
                    if isinstance(answers, list) and len(answers) > 0:
                        ans_block = answers[0]

                    # Case 2: answers is a dict
                    elif isinstance(answers, dict):
                        ans_block = answers
                    else:
                        ans_block = None

                    if ans_block and "answer" in ans_block:
                        answer_items = ans_block["answer"]

                        if isinstance(answer_items, list) and len(answer_items) > 0:
                            answer = answer_items[0]

                            if not answer.get("unanswerable", False):
                                evidence = answer.get("evidence", []) or []
                                extractive_spans = answer.get("extractive_spans", []) or []

                questions.append(
                    {
                        "paper_id": pid,
                        "question_id": qid,
                        "question": qtext,
                        "evidence": evidence,
                        "extractive_spans": extractive_spans,
                    }
                )

        return questions

