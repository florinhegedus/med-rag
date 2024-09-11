import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Raspunde la intrebare, bazandu-te pe urmatorul context:

{context}

---

Raspunde la intrebare, bazat pe contextul de mai sus: {question}
"""


def main():
    load_dotenv()

    st.title("Ajutor pentru REZI")

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    client = OpenAI()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("What is up?"):
        if len(st.session_state.messages) == 0:
            # Search the DB.
            results = db.similarity_search_with_score(user_prompt, k=5)

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            # print(context_text)
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=user_prompt)
        else:
            prompt = user_prompt

        print(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
