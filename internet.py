import urllib2
hasInternet = None
def On():
    global hasInternet
    if hasInternet is None:
        try:
            response=urllib2.urlopen('http://google.com',timeout=1)
            hasInternet = True
        except urllib2.URLError as err:
            hasInternet = False
    return hasInternet
if __name__ == "__main__":
    print On()
