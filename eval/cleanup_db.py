"""清理 knowledge_chunks 中的噪声数据。"""
import asyncio
import re
from app.db.database import async_session_factory
from sqlalchemy import text

_HAN = re.compile(r'[一-鿿]')


def _has_chinese(s: str) -> bool:
    return bool(_HAN.search(s))


async def cleanup():
    async with async_session_factory() as session:
        # 1. 极短垃圾 (<30 chars)
        r1 = await session.execute(
            text('DELETE FROM knowledge_chunks WHERE LENGTH(original_text) < 30')
        )
        print(f'删除极短垃圾 (<30 chars): {r1.rowcount}')

        # 2. 无中文短片段 (30-49 chars, pure code/config)
        r2 = await session.execute(
            text("SELECT id, original_text FROM knowledge_chunks WHERE LENGTH(original_text) BETWEEN 30 AND 49")
        )
        to_delete = []
        for row in r2.fetchall():
            if not _has_chinese(row[1]):
                to_delete.append(row[0])

        if to_delete:
            ids_str = ','.join(str(x) for x in to_delete)
            r3 = await session.execute(
                text(f'DELETE FROM knowledge_chunks WHERE id IN ({ids_str})')
            )
            print(f'删除无中文短片段 (30-49 chars): {r3.rowcount}')

        # 3. 残句开头 (lowercase ascii start, 30-99 chars)
        r4 = await session.execute(
            text("SELECT id, original_text FROM knowledge_chunks WHERE LENGTH(original_text) BETWEEN 30 AND 99")
        )
        to_delete2 = []
        for row in r4.fetchall():
            txt = row[1].strip()
            if txt and txt[0].islower() and txt[0].isascii():
                to_delete2.append(row[0])

        if to_delete2:
            ids_str2 = ','.join(str(x) for x in to_delete2)
            r5 = await session.execute(
                text(f'DELETE FROM knowledge_chunks WHERE id IN ({ids_str2})')
            )
            print(f'删除残句开头 (30-99 chars): {r5.rowcount}')

        r6 = await session.execute(text('SELECT COUNT(*) FROM knowledge_chunks'))
        print(f'\n清理后剩余: {r6.fetchone()[0]} chunks')

        await session.commit()
        print('已提交。')


if __name__ == '__main__':
    asyncio.run(cleanup())
