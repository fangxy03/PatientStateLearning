import re


class EvidenceParser:
    """
    Parse LLM specialist agent output
    into structured evidence representation
    """


    def __init__(self):
        pass



    def parse(self, text):

        result = {

            "state": None,

            "evidence": [],

            "confidence": 0.0,

            "raw": text

        }


        # =========================
        # 1. Extract confidence
        # =========================

        confidence_patterns = [

            r'(\d+)/5',

            r'Confidence.*?(\d)',

            r'confidence.*?(\d)'

        ]


        for pattern in confidence_patterns:

            match = re.search(
                pattern,
                text,
                re.I
            )

            if match:

                score = int(
                    match.group(1)
                )

                result["confidence"] = score / 5

                break



        # =========================
        # 2. Extract state
        # =========================

        state_patterns = [

            r'State[:：]\s*(.*)',

            r'state[:：]\s*(.*)',

        ]


        for pattern in state_patterns:

            match = re.search(
                pattern,
                text
            )

            if match:

                result["state"] = (
                    match.group(1)
                    .strip()
                    .split("\n")[0]
                )

                break



        # =========================
        # 3. Extract evidence
        # =========================

        lines = text.split("\n")


        for line in lines:

            line=line.strip()


            if (

                line.startswith("-")

                or

                line.startswith("•")

            ):

                evidence=line[1:].strip()


                if len(evidence)>5:

                    result["evidence"].append(
                        evidence
                    )


        return result