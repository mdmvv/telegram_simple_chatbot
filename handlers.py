from aiogram import types
from loader import bot
from loader import dp
from chat_model import chat_model


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!")


@dp.message_handler()
async def response(message: types.Message):
    await message.answer(chat_model.generate_response(message.text))
