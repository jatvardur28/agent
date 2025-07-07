sudo systemctl restart telegram-bot.service


# Перезагрузите systemd
sudo systemctl daemon-reload

# Включите автозапуск
sudo systemctl enable telegram-bot.service

# Запустите службу
sudo systemctl start telegram-bot.service

# Проверьте статус
sudo systemctl status telegram-bot.service

# Посмотреть логи
sudo journalctl -u telegram-bot.service -f
