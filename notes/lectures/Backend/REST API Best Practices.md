https://habr.com/ru/articles/351890/
1. **Конечные точки в URL – имя существительное, не глагол**
2. **Множественное число**
3. **Документация**
   Документирование программного обеспечения является общей практикой для всех разработчиков. Этой практики стоит придерживаться и при реализации REST приложений. 
   Наиболее распространенным способом документирования REST приложений – это документация с перечисленными в ней конечными точками, и описывающая список операций для каждой из них. Есть множество инструментов, которые позволяют сделать это автоматически. Например, [Swagger](https://swagger.io/).
4. **Версия вашего приложения**
5. **Пагинация**
   Отправка большого объема данных через HTTP не очень хорошая идея. Безусловно, возникнут проблемы с производительностью, поскольку сериализация больших объектов JSON станет дорогостоящей. Best practice является разбиение результатов на части, а не отправка всех записей сразу. Предоставьте возможность разбивать результаты на странице с помощью предыдущих или следующих ссылок.
6. **Использование SSL**
7. **HTTP методы**
8. **Эффективное использование кодов ответов HTTP**