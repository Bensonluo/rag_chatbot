# PostgreSQL 备份与恢复

此流程恢复 PostgreSQL 中的持久状态。它不自动切换流量、不覆盖现有数据库，也不是整个客服系统的灾备完成证明。

## 选择正确的部署

仓库的开发配置是 `docker-compose.yml`（数据库服务 `postgres`，应用服务 `api`）；生产配置是 `docker-compose.prod.yml`（数据库服务 `db`，应用服务 `app`）。备份、恢复和部署必须使用相同的 Compose 文件、环境文件、项目目录及项目名。运行下面示例前，在项目根目录配置环境文件；如果现有部署指定了 `-p`，设置相同的 `COMPOSE_PROJECT_NAME`。使用外部 Compose 文件时，按原部署设置 `PROJECT_DIRECTORY`；独立备份/恢复脚本未设置该变量时保持 Compose 的文件目录默认行为。

```bash
export COMPOSE_FILE="$PWD/docker-compose.prod.yml"
export PROJECT_DIRECTORY="$PWD"
export ENV_FILE="$PWD/.env.production"
export DB_SERVICE=db
export BACKUP_DIR="$PWD/backups/production"
# 若原部署指定了项目名，在此设置相同的 COMPOSE_PROJECT_NAME。

bash deployment/production/scripts/backup.sh 7
```

数据库用户名和数据库名取自运行中的数据库容器 `POSTGRES_USER` / `POSTGRES_DB`，不靠宿主机未加载的环境文件推测。这个入口备份的是该容器初始化时配置的数据库；如果应用 `DATABASE_URL` 指向其他服务器或数据库，应对那个实际数据库安排独立备份，不能把本命令的成功视作它已被备份。

`backup.sh` 默认仍选择开发配置。生产定时任务应显式设置上述变量，并为各部署使用独立的 `BACKUP_DIR`。

## 备份的成功条件

- `pg_dump` 成功、输出非空、压缩与完整性校验成功后，才在同一文件系统原子发布 `db_backup_<时间>_<随机串>.sql.gz`。文件权限为 0600；新建目录为 0700。
- 同一秒的并发运行使用不同文件名。失败不发布归档，也不执行保留期清理。`pg_dump` 的失败诊断单独保存为 0600 的 `.error.log`，控制台仅打印其路径。
- 保留天数必须为 `0..999999` 的整数；无效值在数据库访问及清理前拒绝。清理只作用于指定目录的直接子文件，遵守 `find -mtime +N` 的完整日龄语义，不递归删除部署快照。
- 本地磁盘上的备份仍会受到同一主机故障影响。按实际 RPO 将已完成的归档及所需配置复制到受控异地存储，并定期重新执行恢复验证。压缩完整性不等于来源真实性；只恢复自己可信来源的 SQL 归档。

## 恢复到新数据库

```bash
# 将路径替换为需要恢复的完整归档；保持上述 Compose/环境/服务选择一致。
bash deployment/production/scripts/restore.sh \
  /secure-backups/db_backup_example.sql.gz \
  --database rag_recovery_20261017
```

目标必须显式提供，使用不超过 63 个 ASCII 字符的简单数据库标识符。`RESTORE_DB` 环境变量也可提供目标。脚本先完整校验并解压到私有临时文件，再通过 `createdb --template=template0` 创建新库；即使目标库为空，只要已存在就拒绝。不会删除、清空或覆盖源库，也不会修改应用连接配置。

这是单数据库逻辑备份，不包含集群级角色或表空间定义。在替换服务器上恢复前，须先准备兼容的 PostgreSQL 版本、归档使用的角色及扩展依赖，并验证权限；本轮实测是在隔离 PostgreSQL 16 实例内恢复到新库，尚未证明整台服务器丢失后的完整重建流程。

随后使用 `psql -X --set=ON_ERROR_STOP=1 --single-transaction --file=-` 执行恢复。普通 SQL 错误会回滚该事务，保留新建库供检查；连接中断时提交状态可能不确定，因此任何失败都不能把目标直接投入服务。脚本不自动删除失败目标。不得把含自行添加 `COMMIT`、切换连接等命令的任意 SQL 当作本工具生成的可信 `pg_dump` 归档。

这些选择遵循 PostgreSQL 的 [SQL dump 恢复说明](https://www.postgresql.org/docs/16/backup-dump.html)及 [psql 的错误停止和单事务选项](https://www.postgresql.org/docs/16/app-psql.html)。

## 切换前验证

先核对恢复库的 Alembic 版本与待运行代码兼容。验证会话和消息内容、引用中的原版本、工单状态与归属、用户事实、缺口处理记录、文档原始字节/归一化内容，以及恢复后新增记录的序列。检查活动工单唯一约束和外键；不要只比较文件大小或表数量。

如果 checkpoint 与业务表同库，确认 LangGraph 能读取已有对话状态和 pending writes。切换期间暂停写入及后台任务，协调应用数据库地址、checkpoint 地址和外部知识库的恢复点，再重建/失效旧缓存。恢复快照之后发生的写入不会自动出现在新库；此次实现没有增量追平、PITR 或自动切流功能。

| 数据 | 本 PostgreSQL 归档的边界 |
|---|---|
| 用户、会话、消息及其引用、工单、用户事实、知识缺口处理 | 所选数据库内的表会备份；恢复演练逐行验证 |
| 文档版本、原始二进制、归一化文本、处理配置 | 位于 PostgreSQL；演练验证字节与内容保持 |
| LangGraph checkpoint | 同库时一起备份；`DIALOGUE_CHECKPOINTER_DB_URL` 指向其他库时需另行备份并协调恢复点 |
| Qdrant 检索片段/向量、Neo4j 图与社区数据 | 独立数据存储，不包含在 SQL 包中；需要各自快照或经过验证的重建流程 |
| commerce-provider 购物车/报价/订单/付款回执 | 独立 SQLite 卷 `commerce_provider_data`，不在 PostgreSQL 归档中；需在停写或通过 SQLite online backup/`.backup` 生成一致快照（不要只复制运行中的 db/wal 文件），并与主库恢复点协调 |
| FAQ、事实表、自定义文件及模型配置 | 保存对应版本的代码和自定义文件；不依赖 SQL 包恢复这些内容 |
| Redis | 主要为 TTL 缓存、限流和临时状态；不能用旧缓存掩盖数据库/知识库恢复缺口 |

已完成的文档任务恢复后不会自动重建被删除的 Qdrant/Neo4j 数据；现有 worker 只领取待处理任务，retry 也有状态限制。旧版无原始文件登记的索引更不能从 PostgreSQL 凭空重建。外部检索库的恢复尚未在本轮完成，不能只恢复 SQL 就宣称完整客服服务已恢复。

## 部署及更新失败

`deploy.sh [dev|prod]` 与 `update.sh [prod|dev]` 在更改服务前保存受保护的部署快照，包含环境配置、Compose 配置、Git 版本、旧容器配置及镜像 ID。快照有凭据，只应由运维访问；不要上传到公开日志。备份失败则停止，不继续拉取、构建或替换应用。

有运行中的应用容器时，部署入口还会读取其有效配置，核验应用数据库及 PostgreSQL checkpoint 指向所选数据库容器：库名、实际端口、容器别名和 DNS 地址均需匹配，连接参数不能另行覆盖目标。外部数据库、独立 checkpoint 库、停止的应用或无法验证的配置会阻止更新，需要独立的一致备份方案。不同用户名访问同一目标库允许通过。没有应用容器时只标注“本地数据库备份已验证”，不推断任何外部数据也已备份。

生产首次部署在没有数据库容器时需要显式 `ALLOW_INITIAL_DEPLOY=1`；没有容器并不能证明没有既存数据卷。已有但停止的数据库不能跳过备份。更新必须有可运行且备份成功的数据库。

检查在选定的容器内执行：PostgreSQL 就绪、应用实际 Redis 连接和应用 `/ready`。其中 `/ready` 表示对话图初始化完成，不是全部下游服务的实时健康证明。超时或更新失败返回非零，保留诊断；不会通过 `down/up` 同一镜像冒充回滚，也不承诺单实例容器替换期间零停机。真正回滚需要结合旧镜像、迁移兼容性、数据库和外部索引恢复点制定方案。

## 可重复演练

```bash
RAG_RUN_RESTORE_DRILL=1 .venv/bin/pytest --no-cov -q \
  app/tests/integration/test_backup_restore_drill.py
```

演练创建随机项目名、随机回环端口的独立 PostgreSQL 16 容器，执行真实迁移、写入业务及 checkpoint 数据、调用上述两个脚本并逐行比较恢复结果；同时验证现有库拒绝、损坏压缩包拒绝、SQL 失败回滚及序列/约束。结束后清理自己创建的容器及卷。它不使用配置的业务数据库，不需要真实模型或真实用户数据。CI 已加入相同演练；远程 CI 是否运行应另查实际任务结果。
