"""Inject chart data into template.html -> snow-to-flow.html."""
t = open("template.html").read()
t = t.replace("/*DATA*/", open("chart_data.json").read()).replace("/*DROUGHT*/", open("drought_data.json").read())
open("snow-to-flow.html", "w").write(t)
