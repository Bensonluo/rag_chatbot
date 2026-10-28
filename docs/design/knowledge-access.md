# 共享知识的访问权限

2026-10-28，本地第22轮合同，尚未部署。知识运营与客户咨询使用不同权限；`DEMO_MODE` 不授予知识管理角色。

## 角色与入口

| 能力 | 公开匿名 Demo | 普通 active 客户 | active admin |
| --- | --- | --- | --- |
| 未标记共享 Demo 聊天 | 沿用现有共享体验 | 可用 | 可用 |
| 私人会话聊天/历史 | 需登录 | 仅自己的会话 | 同样需拥有该会话；获派工单另有授权接口 |
| `/documents/search` | 需登录 | 可检索合资格的共享知识 | 可用 |
| 文档目录、版本列表和状态 | 401 | 403 | 可用 |
| 原始文件、规范化全文 | 401 | 403 | 可用 |
| 上传、同 ID 替换、重试、删除 | 401 | 403 | 可用 |

表中知识管理结果在 Demo 与 strict 模式相同。停用账号为403，无效/过期凭据仍按既有认证流程拒绝。管理需要实际数据库用户的 `is_admin=True`；注册不产生管理权限，角色变更由可信运营流程完成，不能通过请求 body/query 或 JWT 自填角色绕过。

管理入口包含：POST `/api/v1/documents/upload`、`/upload/file`，GET `/api/v1/documents`，GET `/versions/{version_id}`、`/versions/{version_id}/source`、`/versions/{version_id}/text`、`/{document_id}/versions`，POST `/versions/{version_id}/retry`，DELETE `/{document_id}`。同 document_id 上传仍代表替换，没有单独的客户替换许可。

每条管理路由先通过 `get_document_manager` 的 active-user 和管理员检查，再构造业务依赖或进入正文处理；handler 也保留管理检查。拒绝时不能调用文档 job/storage/vector 读写或把新版本排队。FastAPI 对 JSON/multipart 的请求体解析可能先于依赖执行，本合同不宣称拒绝请求零读 HTTP body，也不改变现有上传限制。

`/documents/search` 保留 active 用户的只读检索，仍遵守[知识发布资格](knowledge-publication.md)；普通客户不能通过管理入口取得 queued/failed 原文、全文或全量版本元数据。引用片段可随回答展示，不是完整原始文件的下载授权。`/documents/health` 仍是原有健康端点。Graph 导入及社区检测继续使用既有 admin 依赖，图查询权限及 CLI/worker 可信运行边界不因此改变。

## 身份与缓存

Demo 模式只保留未标记共享会话的匿名聊天/历史行为；新私人会话 owner 规则、知识管理角色和 QA 管理权限在两模式都执行。提供 JWT 不等于管理员权限：每个请求重新从数据库取得 active 用户及当前角色。

文档路由复用 `PrivateAPIRoute`，成功响应、鉴权及可处理的输入验证错误都带 `Cache-Control: private, no-store`，包括管理原文和全文。错误状态、认证头和响应类型保留。此前已下载到客户端的内容不可能由后续权限变化瞬间撤回。

这是当前单一共享知识库的管理角色边界。尚未实现组织/租户、文档 owner 或企业检索 ACL，不能以 admin-only 管理宣称多租户隔离。运营者把客户可检索的政策发布进共享库时仍需自行确认适用范围；本轮没有接通真实业务系统。

## 验证与升级

修改前真实本地注册得到 is_admin=false，Demo 下仍能读未发布原文件、替换管理员文档并经实际 worker/local Qdrant 激活、再删除全部修订；同客户 strict 原文为403。本轮收紧统一角色检查；测试使用真实 register/login/JWT、外键 SQLite、DocumentJobService、文件数据与本地 Qdrant，并对业务方法做记录后原样委托。权限拒绝检查业务调用为零，保留 admin 正向流程与客户只读检索对照。

无需新迁移或依赖。全部提供知识管理 HTTP 的应用实例需升级后，才可宣称部署后的两模式管理边界生效；旧版仍可能按 Demo 放行普通账号。升级不能撤回之前下载的文件，也不会自动将旧普通账号提升为管理员。

具体数量、模式/身份矩阵和源冻结见[第22轮验证结果](../reviews/knowledge-access-results-2026-10-28.json)。本地 SQLite/Qdrant 结果不替代生产、真实 PostgreSQL、多副本、浏览器或真实模型验收。
