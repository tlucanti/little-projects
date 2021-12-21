# /usr/bin/python3.8

##
#	Author:		antikostya
#	Created:	2021-12-13 19:45:31
#	Modified:	2021-12-13 20:09:21
##

from aiohttp import web

routes = web.RouteTableDef()

@routes.view('/serv/main')
class MainHandler(web.View):
	"""

	"""

	async def get(self) -> web.Response:
		# print(dir(self.request))
		content = self.request.text()
		print(dir(content))
		print(content.cr_await())

	async def post(self) -> web.Response:
		content = await self.request.json()
		print(content)


app = web.Application()
app.add_routes(routes)

def main():
	web.run_app(app, port=8080, host='0.0.0.0')

if __name__ == '__main__':
	main()
