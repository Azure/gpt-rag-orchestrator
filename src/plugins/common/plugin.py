from datetime import datetime

from agent_framework import ai_function


class CommonPlugin:

    @ai_function(
        name="GetTodayDate",
        description="The output is today's date in string format",
    )
    def get_today_date(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {today}"

    @ai_function(
        name="GetTime",
        description="The output is the current time (hour and minutes) in HH:MM format",
    )
    def get_time(self) -> str:
        current_time = datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}"