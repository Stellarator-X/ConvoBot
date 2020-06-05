import aiml
import text_prc as tp 
import re
import os
import datetime
import pandas as pd

userdata = pd.read_csv("data/userdata.csv")

kernel = aiml.Kernel()

if os.path.isfile("Brains/bot_brain.brn"):
    kernel.bootstrap(brainFile = "Brains/bot_brain.brn")
else:
    kernel.bootstrap(learnFiles = "aiml/std-startup.aiml", commands = "load")
    kernel.saveBrain("Brains/bot_brain.brn")

# Get session info as dictionary. Contains the input
# and output history as well as any predicates known
sessionId = 1
sessionData = kernel.getSessionData(sessionId)

def getResponse(command, sessionId):
    global kernel, userdata
    if command == "quit" :
        getResponse('save', sessionId)
        kernel.saveBrain('Brains/bot_brain.brn')
        os._exit(1)
    elif command == 'save':
        kernel.saveBrain('Brains/bot_brain.brn')
        # Saving the predicates in the csv file
        userdict = {"Name":kernel.getPredicate("Name", sessionId), 
                        "BookIssued":kernel.getPredicate("BookIssued", sessionId), 
                        "IssueDate":kernel.getPredicate("IssueDate", sessionId), 
                        "DueDate":kernel.getPredicate("DueDate", sessionId)}    
        
        if str(sessionId) in list(userdata['Name']):
            print("Updating records")
            idx = list(userdata['Name']).index(sessionId)
            userdata.loc[idx] = [userdict[key] for key in userdict]
        else:
            userdata = userdata.append(userdict, ignore_index = True)
        
        userdata.to_csv("data/userdata.csv", index = False)

        return "Saved"

    else:
        command = str(tp.get_gist(command))
        response = kernel.respond(command, sessionId)
        if 'getlink' in response:
            w = response.split()
            keywords = ""
            for w_ in w[1:]:
                keywords += "+" + w_
            keywords = keywords[1:]
            response = ("<a href = http://libgen.is/search.php?req="+keywords+"&lg_topic=libgen&open=0&view=simple&res=25&phrase=1&column=def>Here</a>")
            response = "Issuing books has been outsourced to Library Genesis. \nYou can find your book " + response + "(Open this in a new tab)"
            
            duedatestr = (datetime.datetime.now()+datetime.timedelta(days=7)).strftime("%c")
            kernel.setPredicate("DueDate", str(duedatestr), sessionId)
            getResponse('save', sessionId)
        
        elif 'getcatlink' in response:
            w = response.split()
            keywords = ""
            for w_ in w[1:]:
                keywords += "+" + w_
            keywords = keywords[1:]
            pklink = ("<a href = http://libserv.iitk.ac.in/cgi-bin/koha/opac-search.pl?idx=&q="+keywords+"&item_limit=>Here</a>")
            response = "You can find the search results " + pklink + "(Open this in a new tab)"
        
        elif response == "Your book has been renewed.":
            duedatestr = (datetime.datetime.now()+datetime.timedelta(days=7)).strftime("%c")
            kernel.setPredicate("DueDate", duedatestr, sessionId)
            getResponse('save', sessionId)


        sents  = str(response).split('\\n')
        resp = ""
        for line in sents:
            resp += line + "<br></br>" 
        resp = resp[:-9]
        return resp

def main():
    while True:
        command = input("USER : ")
        response = getResponse(command, sessionId) 
        print(f"BOT : {response}\n")

if __name__ == "__main__":
    main()