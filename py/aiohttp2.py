#!/usr/bin/env python3

from aiohttp import web
from json.decoder import JSONDecodeError

routes = web.RouteTableDef()
database = dict()


class ValidationError(ValueError):
    def __init__(self, s):
        super().__init__(s)


def debug_print(*args):
    print(*args)


def bad_request():
    return web.json_response({'message': 'Validation Failed', 'code': 400}, status=400)


def ok_response():
    return web.json_response({}, status=200)


def not_found():
    return web.json_response({'message': 'Item not found', 'code': 404}, status=404)


def node_found(node):
    return web.json_response(node, status=200)


def json_validation(data, expected):
    # if isinstance(expected, list):
    #     assert len(expected) == 1
    #     if not isinstance(data, list):
    #         raise ValidationError(f'expected list of items, got {type(data)}')
    #     for item in data:
    #         json_validation(item, expected[0])
    # else:
        for ekey in expected:
            if ekey.startswith('[') and ekey.endswith(']'):
                dkey = ekey[1:-1]
                if dkey not in data:
                    continue
            else:
                dkey = ekey
            if dkey not in data:
                raise ValidationError(f'key `{dkey}` not in data')
            if isinstance(expected[ekey], (type, tuple)):
                if not isinstance(data[dkey], expected[ekey]):
                    raise ValidationError(f'expected {int} type by key `{dkey}`')
            elif isinstance(expected[ekey], (list, dict)):
                if type(expected[ekey]) != type(data[dkey]):
                    raise ValidationError(f'expected {type(expected[ekey])} type by key `{dkey}`')
                json_validation(data[dkey], expected[ekey])
            else:
                raise ValidationError(f'unknown type by value `{dkey}` ({type(expected[ekey])})')


nonetype = type(None)
import_expected_json = {
    "items": list,
    "updateDate": str
}
category_expected_json = {
    "id": str,
    "name": str,
    "[parentId]": (str, nonetype),
    "[price]": nonetype,
    "type": str,
}
offer_expected_json = {
    "id": str,
    "name": str,
    "[parentId]": (str, nonetype),
    "price": int,
    "type": str,
}


@routes.view('/imports')
class ImportHandler(web.View):
    """

    """
    async def post(self) -> web.Response:
        try:
            content = await self.request.json()
            debug_print('got content:', content)
            json_validation(content, import_expected_json)
            for item in content['items']:
                if 'type' not in item:
                    raise ValidationError('key `type` not in data')
                elif item['type'] == 'CATEGORY':
                    json_validation(item, category_expected_json)
                elif item['type'] == 'OFFER':
                    json_validation(item, offer_expected_json)
                else:
                    raise ValidationError(f'unknown type `{item[type]}`')
                children = dict()
                if item['id'] in database:
                    children = database[item['id']]['children']
                database[item['id']] = item
                database[item['id']]['children'] = children
                if 'parentId' in item and item['parentId'] is not None:
                    database[item['parentId']]['children'][item['id']] = item
            return ok_response()
        except (JSONDecodeError, ValidationError) as exc:
            debug_print('bad request:', exc)
            return bad_request()


@routes.view('/delete/{id}')
class DeleteHandler(web.View):
    """

    """
    async def delete(self) -> web.Response:

        def req_del(item):
            for it in item['children']:
                req_del(item[it])
                del item[it]

        del_id = self.request.match_info.get('id')
        if del_id is None:
            return bad_request()
        if del_id not in database:
            return not_found()

        req_del(database[del_id])
        del database[del_id]
        return ok_response()


@routes.view('/nodes/{id}')
class NodesHandler(web.View):
    """

    """
    async def get(self) -> web.Response:
        node_id = self.request.match_info.get('id')
        if node_id is None:
            return bad_request()
        if node_id not in database:
            return not_found()
        return node_found(database[node_id])

app = web.Application()
app.add_routes(routes)


def main():
    web.run_app(app, port=8080, host='0.0.0.0')


if __name__ == '__main__':
    main()
