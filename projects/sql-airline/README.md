# SQL Airline Project

This project designs and implements a relational airline database system in PostgreSQL that simulates real-world airline operations: aircraft seating layouts, international flight scheduling across time zones, passenger reservations (including group bookings), ticket pricing, and seat assignments.

The database supports both transaction-style operations (reservation → passenger assignment → flight assignment → seat assignment) and analytics queries (profit estimation, seat occupancy reporting, local time conversions).

**Key Features**

**1) Relational Database Schema**

A full normalized schema was created with strong primary/foreign key relationships and integrity constraints. The database includes:

- planes (aircraft details + operating cost per hour)

- seats (seat map per plane, separated by class and row/column rules)

- airport (IATA code + time zones)

- flights (scheduled flights with plane assignments, outbound/return trips)

- plane_seatings (seat inventory per flight + person assigned)

- people (passengers with passport IDs)

- reservations + reservation_passengers (reservation groups)

- ticket_prices (pricing by flight and cabin class)

- reservation_flights (assign reservations to flights)

This structure enforces realistic constraints (example: first class limited to certain rows/columns). 

**2) Automated Seat Layout & Flight Generation**

Instead of manually inserting hundreds of rows, the project uses generate_series() to:

- generate full seat maps for each aircraft

- generate weekly flight schedules between April–December 2025

- correctly handle time zone-aware departure and arrival times (TIMESTAMPTZ)

Routes include:

- Boston (BOS) → London (LHR) → Addis Ababa (ADD) (and return)

- Boston (BOS) ↔ Beijing (PEK) 

**3) Synthetic Passenger + Reservation Simulation**

A realistic passenger population is created using a PostgreSQL function powered by Python Faker (plpython3u) to generate names. 

The system generates:

- ~1200 passengers (passport nation + number)

- reservation groups of varying sizes using a recursive CTE

- reservation codes in realistic airline-style format

**4) Seat Assignment System**

Each reservation is assigned to a flight, and each passenger in that reservation is assigned a seat in plane_seatings.

The pipeline ensures:

- no seat is double-booked

- only empty seats are assigned

- every assigned passenger gets exactly one seat 


**5) Analytics Queries Included**

The project includes SQL queries to answer airline business questions, such as:

- Total profit estimation

  - revenue calculated from occupied seats + ticket prices

  - costs estimated from flight duration × aircraft operating cost 

- Seat occupancy report

  - filled vs remaining seats for a given flight 

  - Local time schedule reporting

  - converts departure/arrival times into origin/destination local time using airport time zones 

 **Tools / Technologies**

PostgreSQL (relational modeling, constraints, recursive CTEs)

Python Faker via plpython3u

generate_series() for scalable data generation

Time zone handling with TIMESTAMPTZ
