from fastapi import FastAPI, HTTPException,Query
from fastapi.responses import StreamingResponse, JSONResponse
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import asyncio
from typing import Any, Dict
import logging
import re
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy import text

app = FastAPI()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化模型和工具
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "vllm-qwen3-32b"

model = ChatOpenAI(
    openai_api_base=VLLM_API_BASE,
    model_name=MODEL_NAME,
    openai_api_key="EMPTY",
    temperature=0.1,
    max_tokens=1024,
    streaming=True
)

#根据你的实际业务场景,更改系统提示词
system_prompt = """你是一个公司业务数据查询小助手,根据用户的提问到数据库中查询业务数据，并给出答案。"""

# 创建代理
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

async def get_available_tools():
    client = MultiServerMCPClient(
        {
            "mysql_mcp": {
                "url": "http://localhost:3003/mcp/",
                "transport": "streamable_http"
            }
        }
    )
    tools = await client.get_tools()
    print(f"获取到的工具: {tools}")
    return tools


# 配置数据库连接 (任选一种)
# MySQL 配置
#connection_string = "mysql+pymysql://用户名:密码@localhost:3306/数据库名字"

# SQLite 配置
connection_string = "sqlite+aiosqlite:///./chat_history.db"  # 当前目录创建数据库

# 创建异步数据库引擎
async_engine = create_async_engine(connection_string)

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        #async_mode=True  # 关键：启用异步模式
    )




async def generate_stream_response(input_data, session_id,show_think):
    '''
    生成流式响应,show_think=True显示思考过程,show_think=False隐藏思考过程
    '''
    tools = await get_available_tools()
    
    # 创建代理执行器
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    # 创建回调处理器
    callback = AsyncIteratorCallbackHandler()
    
    # 添加会话历史支持
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: get_session_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    # 配置会话和回调
    config = {
        "configurable": {"session_id": session_id},
        "callbacks": [callback]
    }
    
    # 异步执行代理
    task = asyncio.create_task(
        agent_with_history.ainvoke(input_data, config=config, timeout=30)
    )

    #显示思考过程
    if show_think:
        try:
        # 直接流式输出每个token
            async for token in callback.aiter():
                logger.info(f"流输出: {token}")
                yield token
        except Exception as e:
            logger.error(f"流输出异常: {str(e)}")
            yield f"【错误】流输出中断: {str(e)}"
        finally:
            await task  # 确保任务完成
            
        logger.info(f"会话 {session_id} 流式响应完成")

    #隐藏思考过程
    if not show_think:
        # 状态机变量
        in_think = False
        buffer = ""
        START_TAG = "<think>"
        END_TAG = "</think>"
        
        # 新增：处理前导空白的标志
        is_leading = True  # 标记是否在响应开头
        skip_whitespace = True  # 是否跳过前导空白
        
        try:
            async for token in callback.aiter():
                # 逐个字符处理
                for char in token:
                    # 新增：跳过前导空白（空格、换行等）
                    if skip_whitespace and char.isspace():
                        continue  # 跳过空白字符
                    
                    # 重置跳过标志（遇到第一个非空白字符后）
                    if skip_whitespace and not char.isspace():
                        skip_whitespace = False
                    
                    if not in_think:
                        # 检查是否可能进入思考标签
                        if char == '<' or (buffer and buffer[0] == '<'):
                            buffer += char
                            
                            # 检查是否完整匹配开始标签
                            if buffer == START_TAG:
                                in_think = True
                                buffer = ""
                                continue
                            # 检查是否部分匹配但无法继续匹配
                            elif not START_TAG.startswith(buffer):
                                # 输出缓冲中不匹配的部分
                                output_chars = buffer[:-1]  # 保留最后一个字符作为新缓冲
                                buffer = buffer[-1]
                                for c in output_chars:
                                    # 新增：跳过思考标签后的前导空白
                                    if is_leading and c.isspace():
                                        continue
                                    is_leading = False
                                    yield c
                        else:
                            # 直接输出非标签内容
                            if buffer:
                                for b in buffer:
                                    # 新增：跳过思考标签后的前导空白
                                    if is_leading and b.isspace():
                                        continue
                                    is_leading = False
                                    yield b
                                buffer = ""
                            # 新增：跳过前导空白
                            if is_leading and char.isspace():
                                continue
                            is_leading = False
                            yield char
                    else:
                        # 在思考标签内部，检查是否结束
                        if char == '<' or (buffer and buffer[0] == '<'):
                            buffer += char
                            
                            # 检查是否完整匹配结束标签
                            if buffer == END_TAG:
                                in_think = False
                                buffer = ""
                                # 新增：标签结束后重置前导空白标志
                                is_leading = True
                            # 检查是否部分匹配但无法继续匹配
                            elif not END_TAG.startswith(buffer):
                                buffer = ""
                        else:
                            # 不是标签开始字符，重置缓冲
                            buffer = ""
        
        except Exception as e:
            logger.error(f"流输出异常: {str(e)}")
            yield f"【错误】流输出中断: {str(e)}"
        finally:
            # 确保任务完成
            try:
                await task
            except Exception as e:
                logger.error(f"代理任务异常: {str(e)}")
            
            # 输出剩余缓冲内容（不在思考标签中）
            if not in_think and buffer:
                for c in buffer:
                    # 新增：跳过前导空白
                    if is_leading and c.isspace():
                        continue
                    is_leading = False
                    yield c
        
        logger.info(f"会话 {session_id} 流式响应完成")
        


# 非流式响应处理
async def get_non_stream_response(input_data: Dict[str, Any], session_id: str, show_think: bool = True) -> str:

    '''
    非流式处理会话,可返回思考过程,也可隐藏思考过程
    show_think: 是否显示思考过程,默认显示,如果设置为False,则显示思考过程
    '''
    logger.info(f"开始非流式处理会话: {session_id}, 输入: {input_data['input']}")

    tools = await get_available_tools()

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,  # 开启详细日志 
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: get_session_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    try:
        # 使用同步调用获取完整响应
        response =  await agent_with_history.ainvoke(
            input_data, 
            config={"configurable": {"session_id": session_id}},
            timeout=30
        )
        logger.info(f"响应内容: {response} ")
        # 提取最终输出
        output = response.get("output", "")
        
        # 隐藏思考标签
        if not show_think:
            cleaned_answer = re.sub(r'<think[^>]*>.*?</think>', '', output, flags=re.DOTALL)
            output = re.sub(r'^\n+', '', cleaned_answer)

        logger.info(f"会话 {session_id} 完成处理")
        
        return output
    
    except asyncio.TimeoutError:
        error_msg = "请求处理超时，请稍后再试"
        logger.warning(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"处理请求时出错: {str(e)}"
        logger.exception(error_msg)
        return error_msg

@app.post("/chat")
async def chat_endpoint(request: Dict[str, Any]):
    user_input = request.get("input")
    session_id = request.get("session_id", "default-session")
    stream = request.get("stream", True)  # 默认为非流式输出
    
    if not user_input:
        raise HTTPException(status_code=400, detail="输入不能为空")
    
    input_data = {"input": user_input}

    #流式输出
    if stream:
        return StreamingResponse(
            # 隐藏思考过程
            generate_stream_response(input_data, session_id,show_think=False)
            , media_type="text/event-stream"
        )

    #非流式输出
    else:
        # 隐藏思考过程
        response = await get_non_stream_response(input_data, session_id,show_think=False)
        return response


@app.get("/sessions")
async def list_sessions():
    TABLE_NAME = "message_store"  # 使用实际表名
    async with AsyncSession(async_engine) as session:
        try:
            # 检查表是否存在 (SQLite 专用方法)
            check_table = text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"
            )
            table_exists = await session.scalar(
                check_table, 
                {"table_name": TABLE_NAME}
            )
            
            if not table_exists:
                return {"sessions": []}  # 表不存在时返回空列表
            
            # 获取所有不同的session_id
            stmt = text(f"SELECT DISTINCT session_id FROM {TABLE_NAME}")
            result = await session.execute(stmt)
            
            sessions = [row[0] for row in result.all()]
            return {"sessions": sessions}
            
        except Exception as e:
            # 记录详细错误
            import logging
            logger = logging.getLogger("uvicorn.error")
            logger.exception("获取会话列表失败")
            
            raise HTTPException(
                status_code=500,
                detail=f"获取会话列表失败: {str(e)}"
            )
@app.get("/session/{session_id}")
async def get_session_history_endpoint(session_id: str):
    # 创建历史记录对象
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        #async_mode=True
    )
    
    # 异步获取消息
    try:
        # 使用 run_in_executor 执行同步方法（aget_messages 是异步的）
        messages = await history.aget_messages()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取历史记录失败: {str(e)}")
    
    return JSONResponse(content={
        "session_id": session_id,
        "messages": [{"type": msg.type, "content": msg.content} for msg in messages]
    })

@app.delete("/session/{session_id}")
async def clear_session_history(session_id: str):
    # 创建历史记录对象
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        #async_mode=True
    )
    
    # 异步获取消息
    try:
        # 使用 run_in_executor 执行同步方法（aclear 是异步的）
        await history.aclear()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除历史记录失败: {str(e)}")
    
    return JSONResponse(content={
        "session_id": session_id,
        "messages": [{"type": '删除', "content":'删除chat记录成功'} ]
    })


@app.delete("/sessions")
async def del_sessions(
    session_id: str = Query(None, description="要删除的会话ID (删除单会话)"),
    all_sessions: bool = Query(False, description="是否删除所有会话")
):
    TABLE_NAME = "message_store"
    
    # 验证参数
    if not session_id and not all_sessions:
        raise HTTPException(
            status_code=400,
            detail="必须提供 session_id 参数或设置 all_sessions=true"
        )
    
    async with AsyncSession(async_engine) as session:
        try:
            # 检查表是否存在
            check_table = text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"
            )
            table_exists = await session.scalar(
                check_table, 
                {"table_name": TABLE_NAME}
            )
            
            if not table_exists:
                return {"deleted": 0, "message": "消息表不存在，无操作"}
            
            # 执行删除操作
            if all_sessions:
                # 删除所有会话
                stmt = text(f"DELETE FROM {TABLE_NAME}")
                result = await session.execute(stmt)
                await session.commit()
                return {
                    "deleted": result.rowcount,
                    "message": f"成功删除所有会话 ({result.rowcount} 条记录)"
                }
                
            elif session_id:
                # 删除特定会话
                stmt = text(
                    f"DELETE FROM {TABLE_NAME} WHERE session_id = :session_id"
                )
                result = await session.execute(stmt, {"session_id": session_id})
                await session.commit()
                
                if result.rowcount == 0:
                    return {
                        "deleted": 0,
                        "message": f"会话 {session_id} 不存在或已被删除"
                    }
                
                return {
                    "deleted": result.rowcount,
                    "message": f"成功删除会话 {session_id} ({result.rowcount} 条记录)"
                }
                
        except Exception as e:
            await session.rollback()
            import logging
            logger = logging.getLogger("uvicorn.error")
            logger.exception(f"会话删除失败: {str(e)}")
            
            raise HTTPException(
                status_code=500,
                detail=f"删除操作失败: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")