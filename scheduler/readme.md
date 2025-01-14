# Scheduler

This is a simple scheduler that will trigger the document fetch logic based on the schedule configuration.

## How it works

The scheduler will check the Cosmos DB for all active schedules. If the schedule is due for execution, it will trigger the document fetch logic.

## How to use

The scheduler is triggered by a timer trigger every day at midnight. The timer trigger is defined in the `function.json` file.

# Cosmos Schedules Schema

```json
{
    "id": "string",
    "lastRun": "string",
    "frequency": "string",
    "companyId": "string",
    "reportType": "string",
    "isActive": "boolean"
}
```

# How to add a new schedule

1. Add a new schedule to the Cosmos DB.
2. The schedule will be active immediately.


