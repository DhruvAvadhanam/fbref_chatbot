from flask import Flask, request, render_template
from dotenv import load_dotenv
import pandas as pd
import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'Premier League_2025')

load_dotenv()

# load the CSVs
prem_standard_df = pd.read_csv(os.path.join(data_dir, 'standard_stats.csv'))
# prem_keeper_df = pd.read_csv(os.path.join(data_dir, 'keeper_stats.csv'))
# prem_defensive_df = pd.read_csv(os.path.join(data_dir, 'defensive_stats.csv'))

context_text = prem_standard_df.to_string(index=False)
# context_text = prem_standard_df.to_json(orient='records', indent=2)

# Define the prompt template
template = """
You are a helpful AI assistant. Answer the user's question based *only* on the provided tabular data context.
If the answer is not in the context, politely say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

# Create PromptTemplate instance
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# set up agent - LLM and Langchain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

llm_chain = LLMChain(prompt=prompt, llm=llm)

def ask_question(prompt):
    response = llm_chain.invoke({"context": context_text, "question": prompt})
    return response["text"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")
    answer = ask_question(user_question)
    return render_template("index.html", question=user_question, response=answer)


if __name__ == '__main__':
    app.run(debug=True)
