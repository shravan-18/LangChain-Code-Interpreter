from langchain.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.tools import PythonREPLTool


def main():
    Code_Query = """Generate 5 QR Codes and save in current working directory, 
                    that point to the address 'https://github.com/shravan-18'"""
    
    CSV_Query = "What were the top 3 movies that got the highest audience score?"

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo"
    )

    python_agent_executer = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    '''
    python_agent_executer.run(
        Code_Query
    )'''

    csv_agent_executer = create_csv_agent(
        llm=llm,
        path="movies.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    csv_agent_executer.run(
        CSV_Query
    )


if __name__ == "__main__":
    main()
