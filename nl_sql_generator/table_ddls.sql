
-- Schema for the synthetic EHR dataset used by the NL→SQL generator.
-- Each CREATE TABLE block below defines an entity that may appear in
-- generated questions. Comments describe the intent of each table to
-- help LLMs reason about the structure.

-- allergies: records allergy episodes for a patient
CREATE TABLE public.allergies (
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text
);
-- careplans: long term treatment plans for a patient
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
-- conditions: diagnoses associated with encounters
CREATE TABLE public.conditions (
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE bigint,
  DESCRIPTION text
);
-- devices: medical devices supplied or implanted during care
CREATE TABLE public.devices (
  STOP timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  UDI text,
  START timestamp with time zone
);
-- encounters: visits between a patient and healthcare providers
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
-- imaging_studies: radiology exams and imaging procedures
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
-- immunizations: vaccinations given to a patient
CREATE TABLE public.immunizations (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  BASE_COST numeric
);
-- medications: prescriptions and administered drugs
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
-- observations: lab results and other measurements
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
-- organizations: hospitals and other healthcare facilities
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
-- patients: demographic information for each person in the dataset
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
-- payer_transitions: yearly insurance coverage changes for a patient
CREATE TABLE public.payer_transitions (
  PATIENT uuid,
  START_YEAR integer,
  END_YEAR integer,
  PAYER uuid,
  OWNERSHIP text
);
-- payers: insurance companies or payment organizations
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
-- procedures: medical procedures performed during encounters
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
-- providers: doctors and other healthcare professionals
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
-- supplies: non‑drug items dispensed to patients
CREATE TABLE public.supplies (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  QUANTITY integer
);
