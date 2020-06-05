from flask import Flask, render_template, request
import bot
from datetime import date

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    global start
    if start:
        start = False
        return get_credentials()
        
    userText = request.args.get('msg')
    return bot.getResponse(userText, bot.sessionId)

@app.route("/get")
def get_credentials():
    # bot.getResponse("load", bot.sessionId)
    name = request.args.get('msg')
    sessionId = name
    bot.sessionId = name
    sessionData = bot.kernel.getSessionData(sessionId)
    bot.kernel.setPredicate("Name", name, bot.sessionId)
    if name in list(bot.userdata['Name']):
        idx = list(bot.userdata['Name']).index(name)
        bot.kernel.setPredicate("newUser", "False", bot.sessionId)
        for predicate in bot.userdata.columns[1:]:
            bot.kernel.setPredicate(predicate, bot.userdata[predicate][idx], bot.sessionId)
    else:
        bot.kernel.setPredicate("newUser", "True", bot.sessionId)
        for predicate in bot.userdata.columns[1:]:
            bot.kernel.setPredicate(predicate, "None", bot.sessionId)
        
    print("Set user.")
    
    return bot.getResponse("boot", bot.sessionId)


if __name__ == "__main__":
    start = True
    app.run()