import asyncio
import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile
import pickle
import os
import re
import matplotlib.pyplot as plt
import io
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import Column, BigInteger, Integer, String, Float, Date, ForeignKey, select, delete, func
from sqlalchemy.orm import DeclarativeBase, relationship, declarative_base
from datetime import date, datetime
from aiogram import BaseMiddleware
import json

# Базовые команды:
## set_profile     - Done
## log_water       - Done
## log_food        - Done
## log_workout     - Done
## check_progress  - Done
# Доп команды
## start_day       - Done


with open('API_KEYS.json', 'r') as file:
    keys = json.load(file)

print(json.dumps(keys, indent=4))

BOT_API_TOKEN = keys['BOT_API_TOKEN'] 
WEATHER_API_KEY = keys['WEATHER_API_KEY'] 
USDA_API_KEY = keys['USDA_API_KEY'] 

class LoggingMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Message, data: dict):
        print(f"Получено сообщение: {event.chat.username} {event.date} {event.text}")
        return await handler(event, data)

# за 1 час за 1 кг веса 
# https://bjuk.ru/tablica-rashoda-kaloriy/
# https://calc-fit.com/gidro#:~:text=%D1%82%D0%B5%D0%BB%D0%B0-,%D0%98%D0%BD%D1%82%D0%B5%D0%BD%D1%81%D0%B8%D0%B2%D0%BD%D0%BE%D1%81%D1%82%D1%8C,%D0%B8%D0%BD%D1%82%D0%B5%D0%BD%D1%81%D0%B8%D0%B2%D0%BD%D0%BE%D1%81%D1%82%D0%B8
Workout_types = {
    'бег':(7, 250), 
    'прогулка':(3, 0), 
    'велосипед':(4, 250), 
    'плаванье':(4, 0), 
    'лыжи':(7, 250), 
    'отжимания':(10, 250)
}


Base = declarative_base()

class Profile(Base):
    __tablename__ = 'profiles'
    tg_id = Column(BigInteger, primary_key=True)
    weight = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    activity = Column(Integer, nullable=False)
    city = Column(String(100), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    days = relationship('Day', back_populates='profile')

class Day(Base):
    __tablename__ = 'days'
    id = Column(Integer, primary_key=True)
    tg_id = Column(BigInteger, ForeignKey("profiles.tg_id"), nullable=False)
    day = Column(Date, nullable=False)
    water_goal = Column(Integer, nullable=False)
    calorie_goal = Column(Integer, nullable=False)

    profile = relationship('Profile', back_populates='days')
    water_logs = relationship('WaterLog', back_populates='day')
    calorie_logs = relationship('CalorieLog', back_populates='day')

class WaterLog(Base):
    __tablename__ = 'water_logs'
    id = Column(Integer, primary_key=True)
    day_id = Column(Integer, ForeignKey("days.id"))
    amount = Column(Integer, nullable=False)
    created_at = Column(Date, default=datetime.utcnow)

    day = relationship('Day', back_populates='water_logs')

class CalorieLog(Base):
    __tablename__ = 'calorie_logs'
    id = Column(Integer, primary_key=True)
    day_id = Column(Integer, ForeignKey("days.id"))
    calories = Column(Integer, nullable=False)
    created_at = Column(Date, default=datetime.utcnow)

    day = relationship('Day', back_populates='calorie_logs')


class Profile_FSM(StatesGroup):
    await_weight = State()
    await_height = State()
    await_age = State()
    await_activity = State()
    await_city = State()


bot = Bot(token=BOT_API_TOKEN)
dp = Dispatcher()
DATABASE_URL = "sqlite+aiosqlite:///db.sqlite3"
dp.message.middleware(LoggingMiddleware())

engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Создание таблиц
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(init_db())

@dp.message(Command('set_profile'))
async def set_profile(message: Message, state: FSMContext):
    await message.reply("Введите ваш вес (в кг):")
    await state.set_state(Profile_FSM.await_weight)

@dp.message(Profile_FSM.await_weight)
async def process_weight(message: Message, state: FSMContext):
    await state.update_data(weight=int(message.text)) # написать экранирование мб
    await message.reply("Введите ваш рост (в см):")
    await state.set_state(Profile_FSM.await_height)

@dp.message(Profile_FSM.await_height)
async def process_height(message: Message, state: FSMContext):
    await state.update_data(height=int(message.text)) # написать экранирование мб
    await message.reply("Введите ваш возраст:")
    await state.set_state(Profile_FSM.await_age)

@dp.message(Profile_FSM.await_age)
async def process_height(message: Message, state: FSMContext):
    await state.update_data(age=int(message.text)) # написать экранирование мб
    await message.reply("Сколько минут активности у вас в день?")
    await state.set_state(Profile_FSM.await_activity)

@dp.message(Profile_FSM.await_activity)
async def process_age(message: Message, state: FSMContext):
    await state.update_data(activity=int(message.text)) # написать экранирование мб
    await message.reply("В каком городе вы находитесь?")
    await state.set_state(Profile_FSM.await_city)

@dp.message(Profile_FSM.await_city)
async def process_age(message: Message, state: FSMContext):
    await state.update_data(city=message.text) # написать экранирование мб
    api_url = f'http://api.openweathermap.org/geo/1.0/direct?q={message.text}&limit={1}&appid={WEATHER_API_KEY}'
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            response = await response.json()
            lat, lon = response[0]['lat'], response[0]['lon']
            await state.update_data(lat=lat, lon=lon)

    await state.update_data(tg_id=message.from_user.id)
    
    data = await state.get_data()
    print(data)
    async with async_session() as session:
        profile = Profile(
            tg_id=data["tg_id"],
            weight=data["weight"],
            height=data["height"],
            age=data["age"],
            activity=data["activity"],
            city=data["city"],
            lat=data["lat"],
            lon=data["lon"],
        )
        session.add(profile)
        await session.commit()
    
    await message.reply('Профиль создан')
    await state.clear()

async def check_temperature(profile):
    api_url = f'https://api.openweathermap.org/data/2.5/weather?lat={profile.lat}&lon={profile.lon}&appid={WEATHER_API_KEY}&units=metric'
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            data = await response.json()
            temp = data['main']['temp_max']
            return temp
    return None

async def calculate_water_goal(profile):
    temp = await check_temperature(profile)
    return int((profile.weight * 30 + 500 * profile.activity/30) * (1 + .1 * (temp > 25))), temp

async def calculate_calorie_goal(profile):
    # Минимальная (1.2): сидячая работа, отсутствие спорта;
    # Легкая (1.35): легкие физические упражнения около 3 раз за неделю, ежедневная утренняя зарядка, пешие прогулки;
    # Средняя (1.55): спорт до 5 раз за неделю;
    # Высокая (1.75): активный образ жизни вкупе с ежедневными интенсивными тренировками;
    # Экстремальная (1.95): максимальная активность - спортивный образ жизни, тяжелый физический труд, длительные тяжелые тренировки каждый день.
    # https://www.yournutrition.ru/calories/#:~:text=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B5%D1%86%D0%B8%D0%B5%D0%BD%D1%82%D1%8B%20%D0%BD%D0%B0%D0%B3%D1%80%D1%83%D0%B7%D0%BA%D0%B8%3A
    if profile.activity < 10:
        coeff = 1.2
    elif profile.activity < 45:
        coeff = 1.35
    elif profile.activity < 90:
        coeff = 1.55
    elif profile.activity < 120:
        coeff = 1.75
    else:
        coeff = 1.95

    return int(coeff*(10 * profile.weight + 6.25 * profile.height - 5 * profile.age))

@dp.message(Command('start_day'))
async def start_day(message: Message):
    async with async_session() as session:
        profile = await session.get(Profile, message.from_user.id)
        if not profile:
            await message.reply("Сначала создайте профиль")
            return
        water_goal, temp = await calculate_water_goal(profile)
        calorie_goal = await calculate_calorie_goal(profile)

        day = await session.scalar(
            select(Day).where(
                Day.tg_id == profile.tg_id,
                Day.day == date.today()
            )
        )

        if day:
            await session.execute(
                delete(WaterLog).where(WaterLog.day_id == day.id)
            )
            await session.execute(
                delete(CalorieLog).where(CalorieLog.day_id == day.id)
            )
            await session.delete(day)
            await session.commit()

        new_day = Day(
            tg_id=profile.tg_id,
            day=date.today(),
            water_goal=water_goal,
            calorie_goal=calorie_goal
        )

        session.add(new_day)
        session.add(WaterLog(day_id=new_day.id, amount=0))
        session.add(CalorieLog(day_id=new_day.id, calories=0))
        await session.commit()

    await message.reply(
        f'День начат.\n'
        f'На улице {temp}°C.\n'
        f'Ваша цель на сегодня:\n'
        f' Вода: {water_goal} мл\n'
        f' Калории: {calorie_goal} ккал'
    )

class ReqRes(StatesGroup):
    req = State()

@dp.message(Command('log_water'))
async def log_water(message: Message):
    amount = int(message.text[len('/log_water '):])

    async with async_session() as session:
        day = await session.scalar(
            select(Day).where(
                Day.tg_id == message.from_user.id,
                Day.day == date.today()
            )
        )
        if not day:
            await message.reply("День не начат")
            return

        session.add(WaterLog(day_id=day.id, amount=amount))
        await session.commit()

        total = await session.scalar(
            select(func.sum(WaterLog.amount)).where(WaterLog.day_id == day.id)
        )
    await message.reply(f'Выпито {total} из {day.water_goal} мл')

async def get_calories(item):
    api_url = f'https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&query={item}&dataType=Foundation&pageSize=1'
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            data = await response.json()
            if len(data['foods']) > 0:
                food = data['foods'][0]
                food_name = food['description']
                for item in food['foodNutrients']:
                    if item['nutrientId'] == 1008:
                        return food_name, item['value']
    return None, 0

@dp.message(Command('log_food'))
async def log_calories_p1(message: Message, state: FSMContext):
    food_name = message.text[len('/log_food '):]
    found_name, calories = await get_calories(food_name)
    if not found_name:
        await message.reply(f'Такого продукта не найдено, попробуйте снова.')
        return
    await state.update_data(calories = calories)
    await state.set_state(ReqRes.req)
    await message.reply(f"Найден {found_name} - {calories} ккал на 100 г. Сколько грамм вы съели?")

@dp.message(ReqRes.req)
async def log_calories_p2(message: Message, state: FSMContext):
    data = await state.get_data()
    amount = int(message.text) * data.get('calories')/100

    async with async_session() as session:
        day = await session.scalar(
            select(Day).where(
                Day.tg_id == message.from_user.id,
                Day.day == date.today()
            )
        )
        if not day:
            await message.reply("День не начат")
            return

        session.add(CalorieLog(day_id=day.id, calories=amount))
        await session.commit()

    await message.reply(f'Записано {amount} ккал.')
    await state.clear()

@dp.message(Command('log_workout'))
async def log_workout(message: Message):
    p = re.compile('log_workout\\s([A-Za-zА-Яа-я ]+)\\s([0-9]+)')
    workout_data = p.findall(message.text)
    workout_type = workout_data[0][0]
    if workout_type.lower() not in Workout_types:
        await message.reply(
            f"Такие занятия пока не внесены в список, выберите одно из: {Workout_types}"
        )
        return
    ccal_per_hour_per_kilo, water_per_hour = Workout_types[workout_type.lower()]
    workout_time = int(workout_data[0][1])

    async with async_session() as session:
        profile = await session.get(Profile, message.from_user.id)
        if not profile:
            await message.reply("Сначала создайте профиль")
            return

        day = await session.scalar(
            select(Day).where(
                Day.tg_id == message.from_user.id,
                Day.day == date.today()
            )
        )
        if not day:
            await message.reply("День не начат")
            return
        burned = int(ccal_per_hour_per_kilo/60 * workout_time * profile.weight)
        to_drink = int(water_per_hour/60 * workout_time)
        session.add(CalorieLog(day_id=day.id, calories=-burned))
        day.water_goal += to_drink

        await session.commit()
    
    await message.reply(f'{workout_type} {workout_time} минут - {burned} ккал. Дополнительно выпейте {to_drink} мл воды.')



@dp.message(Command('check_progress'))
async def check_progress(message: Message):
    async with async_session() as session:
        day = await session.scalar(
            select(Day).where(
                Day.tg_id == message.from_user.id,
                Day.day == date.today()
            )
        )

        if not day:
            await message.reply("День не начат")
            return

        water_rows = (
            await session.execute(
                select(WaterLog.amount)
                .where(WaterLog.day_id == day.id)
                .order_by(WaterLog.created_at)
            )
        ).scalars().all()

        calorie_rows = (
            await session.execute(
                select(CalorieLog.calories)
                .where(CalorieLog.day_id == day.id)
                .order_by(CalorieLog.created_at)
            )
        ).scalars().all()

        water_cumsum = np.cumsum(water_rows) if water_rows else [0]

        eaten = [c for c in calorie_rows if c > 0]
        burned = [-c for c in calorie_rows if c < 0]

        calories_cumsum = np.cumsum(eaten) if eaten else [0]

        bio = io.BytesIO()
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Выпитая вода")
        plt.plot(water_cumsum)
        plt.ylim(0, day.water_goal)

        plt.subplot(1, 2, 2)
        plt.title("Потребление калорий")
        plt.plot(calories_cumsum)
        plt.ylim(0, day.calorie_goal)

        plt.tight_layout()
        plt.savefig(bio, dpi=250, format="png")
        plt.close()

        await message.answer_photo(
            BufferedInputFile(bio.getvalue(), filename="progress.png"),
            caption="Графики прогресса"
        )

        # ---------- текст ----------
        total_water = sum(water_rows)
        total_eaten = sum(eaten)
        total_burned = sum(burned)

        await message.reply(
            "Прогресс:\n"
            "Вода:\n"
            f" - Выпито: {total_water} мл из {day.water_goal}\n"
            f" - Осталось: {max(day.water_goal - total_water, 0)}\n\n"
            "Калории:\n"
            f" - Потреблено: {total_eaten} ккал из {day.calorie_goal}\n"
            f" - Сожжено: {total_burned} ккал\n"
            f" - Баланс: {total_eaten - total_burned}"
        )

# Основная функция запуска бота
async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())