# Настройка оболочки для перехвата ошибок внутри пайпов
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

# Переменная для сообщения по умолчанию
m ?=

.PHONY: gitsync
gitsync:
	@git add .
	@if git diff --cached --quiet; then \
		echo "ℹ️ Нет изменений для коммита."; \
		exit 0; \
	fi
	@# Проверка: передано ли сообщение и не состоит ли оно только из пробелов
	@if [ -n "$(m)" ] && [ -n "$$(echo "$(m)" | tr -d ' ')" ]; then \
		git commit -m "feat: $(m)"; \
	else \
		FILES=$$(git diff --cached --name-only); \
		COUNT=$$(echo "$$FILES" | wc -l); \
		if [ $$COUNT -le 3 ]; then \
			LIST=$$(echo "$$FILES" | tr '\n' ',' | sed 's/ ,//;s/,$$//'); \
			git commit -m "feat: update $$LIST"; \
		else \
			git commit -m "feat: update $$COUNT files"; \
		fi \
	fi
	@CURRENT_BRANCH=$$(git branch --show-current); \
	git push origin "$$CURRENT_BRANCH"
	@echo "🚀 Успешно запушено в branch: $${CURRENT_BRANCH}!"
