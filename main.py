from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, create_csv_agent, initialize_agent
from langchain.tools import PythonREPLTool
from  langchain.tools import Tool
 

def main():
    Code_Query = """Generate 5 QR Codes and save in current working directory, 
                    that point to the address 'https://github.com/shravan-18',
                    you already have all necessary libraries installed"""
    
    CSV_Query = "What were the top 3 movies that got the highest audience score?"

    Grand_Query = CSV_Query

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
    '''
    csv_agent_executer.run(
       CSV_Query
    )'''

    grand_agent = initialize_agent(tools=[
        Tool(
            name="PythonAgent",
            func=python_agent_executer.run,
            description="""useful when you need to transform natural language and write it in the form of Python code
                            , returning the results of execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL"""
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent_executer.run,
            description="""useful when you need to answer questions based on 'movies.csv' file,
                            take input the entire question and returns the result after running Pandas calculations"""
        ),
    ],
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    )

    try:
        grand_agent.run(
            Grand_Query
        )

    except Exception as e:
             response = str(e)
             if response.startswith("Could not parse LLM output: `"):
                  response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
                  print(response)

if __name__ == "__main__":
    main()
