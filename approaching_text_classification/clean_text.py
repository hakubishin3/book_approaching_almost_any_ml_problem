import re
import string


def clean_text(s: str) -> str:
    """
    This function cleans the text a bit
    """
    # remove all punctuations using regex and string module
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return s


if __name__ == "__main__":
    corpus = pd.read_csv("imdb.csv")
    corpus.loc[:, "review"] = corpus["review"].apply(clean_text)
