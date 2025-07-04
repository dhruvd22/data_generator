
CREATE TABLE public.allergies (
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text
);
CREATE TABLE public.careplans (
  Id uuid NOT NULL,
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE bigint,
  DESCRIPTION text,
  REASONCODE bigint,
  REASONDESCRIPTION text,
  CONSTRAINT careplans_pkey PRIMARY KEY (Id)
);
CREATE TABLE public.conditions (
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE bigint,
  DESCRIPTION text
);
CREATE TABLE public.devices (
  STOP timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  UDI text,
  START timestamp with time zone
);
CREATE TABLE public.encounters (
  Id uuid NOT NULL,
  START timestamp with time zone,
  STOP timestamp with time zone,
  PATIENT uuid,
  ORGANIZATION uuid,
  PROVIDER uuid,
  PAYER uuid,
  ENCOUNTERCLASS text,
  CODE bigint,
  DESCRIPTION text,
  BASE_ENCOUNTER_COST numeric,
  TOTAL_CLAIM_COST numeric,
  PAYER_COVERAGE numeric,
  REASONCODE bigint,
  REASONDESCRIPTION text,
  CONSTRAINT encounters_pkey PRIMARY KEY (Id)
);
CREATE TABLE public.imaging_studies (
  Id uuid NOT NULL,
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  BODYSITE_CODE text,
  BODYSITE_DESCRIPTION text,
  MODALITY_CODE text,
  MODALITY_DESCRIPTION text,
  SOP_CODE text,
  SOP_DESCRIPTION text,
  CONSTRAINT imaging_studies_pkey PRIMARY KEY (Id)
);
CREATE TABLE public.immunizations (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  BASE_COST numeric
);
CREATE TABLE public.medications (
  START timestamp with time zone,
  STOP timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  BASE_COST numeric,
  PAYER_COVERAGE numeric,
  DISPENSES bigint,
  TOTALCOST numeric,
  REASONCODE bigint,
  REASONDESCRIPTION text,
  PAYER uuid
);
CREATE TABLE public.observations (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE text,
  DESCRIPTION text,
  VALUE text,
  UNITS text,
  TYPE text
);
CREATE TABLE public.organizations (
  Id uuid NOT NULL,
  NAME text,
  ADDRESS text,
  CITY text,
  STATE text,
  ZIP text,
  LAT numeric,
  LON numeric,
  PHONE text,
  REVENUE numeric,
  UTILIZATION integer,
  CONSTRAINT organizations_pkey PRIMARY KEY (Id)
);
CREATE TABLE public.patients (
  Id uuid,
  BIRTHDATE date,
  DEATHDATE date,
  SSN text,
  DRIVERS text,
  PASSPORT text,
  PREFIX text,
  FIRST text,
  LAST text,
  SUFFIX text,
  MAIDEN text,
  MARITAL text,
  RACE text,
  ETHNICITY text,
  GENDER text,
  BIRTHPLACE text,
  ADDRESS text,
  CITY text,
  STATE text,
  COUNTY text,
  ZIP integer,
  LAT numeric,
  LON numeric,
  HEALTHCARE_EXPENSES numeric,
  HEALTHCARE_COVERAGE numeric
);
CREATE TABLE public.payer_transitions (
  PATIENT uuid,
  START_YEAR integer,
  END_YEAR integer,
  PAYER uuid,
  OWNERSHIP text
);
CREATE TABLE public.payers (
  Id uuid,
  NAME text,
  ADDRESS text,
  CITY text,
  STATE_HEADQUARTERED text,
  ZIP integer,
  PHONE text,
  AMOUNT_COVERED numeric,
  AMOUNT_UNCOVERED numeric,
  REVENUE numeric,
  COVERED_ENCOUNTERS numeric,
  UNCOVERED_ENCOUNTERS numeric,
  COVERED_MEDICATIONS numeric,
  UNCOVERED_MEDICATIONS numeric,
  COVERED_PROCEDURES numeric,
  UNCOVERED_PROCEDURES numeric,
  COVERED_IMMUNIZATIONS numeric,
  UNCOVERED_IMMUNIZATIONS numeric,
  UNIQUE_CUSTOMERS integer,
  QOLS_AVG numeric,
  MEMBER_MONTHS integer
);
CREATE TABLE public.procedures (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE bigint,
  DESCRIPTION text,
  BASE_COST numeric,
  REASONCODE bigint,
  REASONDESCRIPTION text
);
CREATE TABLE public.providers (
  Id uuid,
  ORGANIZATION uuid,
  NAME text,
  GENDER text,
  SPECIALITY text,
  ADDRESS text,
  CITY text,
  STATE text,
  ZIP text,
  LAT numeric,
  LON numeric,
  UTILIZATION bigint
);
CREATE TABLE public.supplies (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  QUANTITY integer
);
