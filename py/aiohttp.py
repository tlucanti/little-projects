#!/usr/bin/env python3

from aiohttp import web
from typing import List
from pydantic import BaseModel, constr
from pydantic import parse_obj_as
from json.decoder import JSONDecodeError
from pydantic import ValidationError
from random import randint

routes = web.RouteTableDef()


class ChatNameBase(BaseModel):
    """
    Base class for chat name content
    """
    chat_name: constr(max_length=255)


class UserNameBase(BaseModel):
    """
    Base class for chat name content
    """
    user_name: constr(max_length=255)


class MessageSchema(BaseModel):
    """
    Base class for message schema
    """
    message: str


class MessageArray(BaseModel):
    """
    Base class for message schema
    """
    messages: List[MessageSchema]


class MessageCursor(BaseModel):
    """
    Base class for message cursor
    """
    iterator: str


def default_error_response(message: str):
    """
    default_error_response def
    """
    return web.json_response({'message': message}, status=500)


def not_found_response(message: str):
    """
    Not found 404 response def
    """
    return web.json_response({'message': message}, status=404)


def bad_parameter_response():
    """
    default_error_response def
    """
    return web.json_response({'message': 'bad-parameters'}, status=400)


def chat_create_response(chat_id: int):
    """
    ChatCreateResponse def
    """
    return web.json_response({'chat_id': 'wqerqe'}, status=201)


def chat_join_response(user_id: int):
    """
    chat_join_response def
    """
    return web.json_response({'user_id': str(user_id)}, status=201)


def chat_send_message_response(message_id: int):
    """
    chat_send_message_response def
    """
    return web.json_response({'message_id': str(message_id)}, status=201)

def chat_get_message_response(messages: MessageArray, next_: [MessageCursor]):
    """
    chat_get_message_response def
    """
    ms = messages.dict()
    ms['next'] = next_.dict()
    return web.json_response(ms, status=200)


class ChatNotFoundException(ValueError):
    def __init__(self, _: int):
        super().__init__('chat not found')


def create_chat(_: str) -> int:
    return randint(1000, 9999)


def get_chat_user_id(_: int, __: str) -> int:
    return randint(1000, 9999)


def get_message_array(_: str, limit: int) -> MessageArray:
    return parse_obj_as(MessageArray, {'messages': [{'message': 'my_message'}] * limit})


def get_next_message(from_: str, limit: int) -> MessageCursor:
    return parse_obj_as(MessageCursor, {'iterator': f'next message({hex(id(from_) + limit)})'})


def get_message_id(_: int, __: str) -> int:
    return randint(1000, 9999)


def print_exc(exc) -> None:
    if isinstance(exc, AssertionError):
        print('wrong data limits')
    elif isinstance(exc, JSONDecodeError):
        print('wrong json data')
        print(exc)
    elif isinstance(exc, ValidationError):
        print('wrong json format')
        print(exc)
    elif isinstance(exc, Exception):
        print('another exception ocured')
        print(exc)
    elif isinstance(exc, int) or isinstance(exc, str):
        print(f'exception: {exc}')
    else:
        print('unknown exception ocured')
        print(exc)
    print()
    print()


@routes.view('/v1/chats')
class ChatsHandler(web.View):
    """
    Handler for "/v1/chats" URLs

    ...

    Methods
    -------
    post(request: web.Request) -> web.Response
        handler for post response
        returns something
    """

    async def post(self) -> web.Response:
        try:
            content = await self.request.json()
            print('got content:', content)
            chat_name_base = parse_obj_as(ChatNameBase, content)
            chat_name = create_chat(chat_name_base.chat_name)
        except (JSONDecodeError, ValidationError) as exc:
            print_exc(exc)
            return bad_parameter_response()
        except Exception as exc:
            print_exc(exc)
            return default_error_response("error reason")
        return chat_create_response(chat_name)


@routes.view('/v1/chats/{chat_id}/users')
class ChatHandler(web.View):
    """
    Handler for "/v1/chats/{chat_id}/users" URLs

    ...

    Methods
    -------
    post(request: web.Request) -> web.Response
        handler for post response
        returns something
    """

    async def post(self) -> web.Response:
        chat_id = self.request.match_info.get('chat_id', None)
        if chat_id is None:
            print_exc(404)
            return bad_parameter_response()
        try:
            content = await self.request.json()
            user_name_base = parse_obj_as(UserNameBase, content)
            chat_user_id = get_chat_user_id(chat_id, user_name_base.user_name)
        except (JSONDecodeError, ValidationError) as exc:
            print_exc(exc)
            return bad_parameter_response()
        except ChatNotFoundException as exc:
            print_exc(exc)
            return not_found_response("chat-not-found")
        except Exception as exc:
            print_exc(exc)
            return default_error_response("error reason")
        return chat_join_response(chat_user_id)


@routes.view('/v1/chats/{chat_id}/messages')
class MessagesHandler(web.View):
    """
    Handler for "/v1/chats/{chat_id}/messages" URLs

    ...

    Methods
    -------
    get(request: web.Request) -> web.Response
        handler for get request
        returns something

    post(request: web.Request) -> web.Response
        handler for post response
        returns something
    """

    async def get(self) -> web.Response:
        chat_id = self.request.match_info.get('chat_id', None)
        limit = (lambda x: None if x == "" else x)(self.request.rel_url.query.get('limit', ""))
        from_ = (lambda x: None if x == "" else x)(self.request.rel_url.query.get('from', ""))
        print('data:', {'chat_id': chat_id, 'limit': limit, 'from': from_})
        if None in (chat_id, limit):
            print_exc('limit not provided')
            return bad_parameter_response()
        try:
            limit = int(limit)
            assert 0 <= limit <= 1000
            message_array = get_message_array(from_, limit)
            next_message = get_next_message(from_, limit)
        except AssertionError as exc:
            print_exc(exc)
            return bad_parameter_response()
        except ChatNotFoundException as exc:
            print_exc(exc)
            return not_found_response("chat-not-found")
        except Exception as exc:
            print_exc(exc)
            return default_error_response("error reason")
        return chat_get_message_response(message_array, next_message)

    async def post(self) -> web.Response:
        chat_id = (lambda x: None if x == "" else x)(self.request.match_info.get('chat_id', ""))
        user_id = (lambda x: None if x == "" else x)(self.request.rel_url.query.get('user_id', ""))
        if None in (chat_id, user_id):
            print_exc('user_id not provided')
            return bad_parameter_response()
        try:
            content = await self.request.json()
            message_base = parse_obj_as(MessageSchema, content)
            message_id = get_message_id(chat_id, message_base.message)
        except (JSONDecodeError, ValidationError) as exc:
            print_exc(exc)
            return bad_parameter_response()
        except Exception as exc:
            print_exc(exc)
            return default_error_response('error reason')
        return chat_send_message_response(message_id)


app = web.Application()
app.add_routes(routes)


def main():
    web.run_app(app, port=8080, host='0.0.0.0')


if __name__ == '__main__':
    main()
