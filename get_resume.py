import asyncio
import asyncpg
import json

async def main():
    try:
        conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5433/agent_memory")
        rows = await conn.fetch("SELECT section, content FROM resume_embeddings")
        results = {}
        for row in rows:
            section = row['section']
            content = row['content']
            if section not in results:
                results[section] = []
            results[section].append(content)
        
        print(json.dumps(results, indent=2))
        await conn.close()
    except Exception as e:
        print(f"Failed to connect: {e}")

asyncio.run(main())
