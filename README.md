### 使用langchain搭建智能体

## 项目介绍:
本项目是一个基于langchain搭建的智能体,可以接入mcp服务器或自制的工具

## 项目架构:
- 后端: 基于fastapi + langchain 搭建的后端接口
- 数据库: sqlite数据库或内存

## 项目运行:
    - 进入项目目录: cd 预定开庭室智能体/后端项目
    - 安装依赖: pip install -r requirements.txt
    - 运行项目: 
      - 智能体(使用内存记忆历史记录,重启后会丢失历史记录)
            uvicorn agent:app --host 0.0.0.0 --port 8005 --reload 
      - 智能体(使用sqlite数据库持久化历史记录)
            uvicorn agent_use_sqlite:app --host 0.0.0.0 --port 8005 --reload 

## 项目文件
- agent.py: 智能体(使用内存记忆历史记录,重启后会丢失历史记录)
- agent_use_sqlite.py: 使用sqlite数据库持久化历史记录的智能体
- README.md: 项目说明文档
- requirements.txt: 项目依赖文件

## api接口:

1.聊天
 POST   http://localhost:8005/chat
 Content-Type: application/json
 请求参数:
 {"input": "你好,你是谁", "session_id": "test123","stream": true}

 
2.查看指定会话历史记录
 Get http://localhost:8005/session/{session_id}

3.查看所有会话历史记录
 Get http://localhost:8005/sessions
 
4.删除指定会话接口
DELETE "http://localhost:8005/sessions?session_id=abc123"

5.删除所有会话
DELETE "http://localhost:8005/sessions?all_sessions=true"


## curl测试api接口

### chat接口测试:
流式输出:
curl -D POST  http://localhost:8005/chat  -H 'content-type: application/json'  -d  '{"input": "你好,你是谁", "session_id": "test123"}'
非流式输出:
curl -D POST  http://localhost:8005/chat  -H 'content-type: application/json'  -d  '{"input": "你好,你是谁", "session_id": "test123","stream": false}'
 
 
curl -D POST  http://localhost:8005/chat  -H 'content-type: application/json'  -d  '{"input": "帮我查询下fjag用户的购买的商品信息", "session_id": "test123"}'
  
### 查看指定session历史聊天接口测试:
curl -D GET http://localhost:8005/session/test123


### 删除指定会话接口测试:
curl -X DELETE "http://localhost:8005/sessions?session_id=abc123"

response:
{
  "deleted": 5,
  "message": "成功删除会话 abc123 (5 条记录)"
}

### 删除所有会话接口测试:
curl -X DELETE "http://localhost:8005/sessions?all_sessions=true"
response:
{
  "deleted": 42,
  "message": "成功删除所有会话 (42 条记录)"
} 
