import { useEffect, useRef, useState } from "react";
import { Upload, Trash2, FileText, Loader2 } from "lucide-react";
import { deleteKnowledge, listKnowledge, uploadKnowledge } from "../lib/api";
import type { KnowledgeListItem } from "../lib/types";

export function KnowledgePanel() {
  const [docs, setDocs] = useState<KnowledgeListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      setDocs(await listKnowledge());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadKnowledge(file);
      alert(`上传成功：${file.name} 共 ${result.chunks_count} 个片段`);
      await refresh();
    } catch (err) {
      alert(`上传失败：${err}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (sourceFile: string) => {
    if (!confirm(`确认删除 ${sourceFile} 的全部知识片段？`)) return;
    setDeletingFile(sourceFile);
    try {
      await deleteKnowledge(sourceFile);
      await refresh();
    } catch (err) {
      alert(`删除失败：${err}`);
    } finally {
      setDeletingFile(null);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      <header className="px-8 py-5 border-b border-cream-300 bg-cream-50/50 backdrop-blur flex items-center justify-between">
        <div>
          <h2 className="font-serif text-xl font-semibold text-ink-700">
            知识库
          </h2>
          <p className="text-xs text-ink-400 mt-0.5">
            上传 Markdown 教材，按标题自动切块并生成向量索引
          </p>
        </div>
        <label className="btn-primary cursor-pointer">
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,.markdown,.txt"
            onChange={handleFileChange}
            className="hidden"
            disabled={uploading}
          />
          {uploading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> 上传中…
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" /> 上传 Markdown
            </>
          )}
        </label>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-4xl mx-auto">
          {loading && (
            <p className="text-center text-ink-400 py-12">
              <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
              加载中…
            </p>
          )}
          {!loading && docs.length === 0 && (
            <div className="text-center py-16 card p-12">
              <FileText className="w-12 h-12 text-ink-400 mx-auto mb-4" />
              <h3 className="font-serif text-xl text-ink-700 mb-2">
                还没有知识库
              </h3>
              <p className="text-sm text-ink-400">
                点击右上角"上传 Markdown"导入第一份教材
              </p>
            </div>
          )}
          {!loading && docs.length > 0 && (
            <div className="grid gap-3">
              {docs.map((doc) => (
                <div
                  key={doc.source_file}
                  className="card p-4 flex items-center gap-4 hover:shadow-warm transition-shadow"
                >
                  <div className="w-10 h-10 rounded bg-terracotta-500/10 grid place-items-center shrink-0">
                    <FileText className="w-5 h-5 text-terracotta-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-ink-700 truncate">
                      {doc.source_file}
                    </div>
                    <div className="text-xs text-ink-400 mt-0.5">
                      {doc.chunks_count} 个片段 ·{" "}
                      {new Date(doc.created_at).toLocaleString("zh-CN")}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.source_file)}
                    disabled={deletingFile === doc.source_file}
                    className="btn-icon hover:text-terracotta-700 hover:bg-terracotta-500/10"
                    title="删除"
                  >
                    {deletingFile === doc.source_file ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
