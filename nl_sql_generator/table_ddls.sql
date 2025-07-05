
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
COMMENT ON COLUMN public."allergies"."START" IS 'Date when the allergy episode began';
COMMENT ON COLUMN public."allergies"."STOP" IS 'Date when the allergy episode resolved';
COMMENT ON COLUMN public."allergies"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."allergies"."ENCOUNTER" IS 'Encounter during which the allergy was recorded';
COMMENT ON COLUMN public."allergies"."CODE" IS 'Code representing the allergy';
COMMENT ON COLUMN public."allergies"."DESCRIPTION" IS 'Human readable allergy description';
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
COMMENT ON COLUMN public."careplans"."Id" IS 'Primary key for the care plan';
COMMENT ON COLUMN public."careplans"."START" IS 'Date the care plan started';
COMMENT ON COLUMN public."careplans"."STOP" IS 'Date the care plan ended';
COMMENT ON COLUMN public."careplans"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."careplans"."ENCOUNTER" IS 'Encounter associated with the plan';
COMMENT ON COLUMN public."careplans"."CODE" IS 'Care plan code';
COMMENT ON COLUMN public."careplans"."DESCRIPTION" IS 'Description of the care plan';
COMMENT ON COLUMN public."careplans"."REASONCODE" IS 'Reason code for the care plan';
COMMENT ON COLUMN public."careplans"."REASONDESCRIPTION" IS 'Reason description for the care plan';
-- conditions: diagnoses associated with encounters
CREATE TABLE public.conditions (
  START date,
  STOP date,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE bigint,
  DESCRIPTION text
);
COMMENT ON COLUMN public."conditions"."START" IS 'Date when the condition was first noted';
COMMENT ON COLUMN public."conditions"."STOP" IS 'Date when the condition resolved';
COMMENT ON COLUMN public."conditions"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."conditions"."ENCOUNTER" IS 'Encounter tied to the condition';
COMMENT ON COLUMN public."conditions"."CODE" IS 'Code for the diagnosis';
COMMENT ON COLUMN public."conditions"."DESCRIPTION" IS 'Text description of the diagnosis';
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
COMMENT ON COLUMN public."devices"."STOP" IS 'Timestamp when the device was removed';
COMMENT ON COLUMN public."devices"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."devices"."ENCOUNTER" IS 'Encounter associated with the device';
COMMENT ON COLUMN public."devices"."CODE" IS 'Device code';
COMMENT ON COLUMN public."devices"."DESCRIPTION" IS 'Description of the device';
COMMENT ON COLUMN public."devices"."UDI" IS 'Unique device identifier';
COMMENT ON COLUMN public."devices"."START" IS 'Timestamp when the device was placed';
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
COMMENT ON COLUMN public."encounters"."Id" IS 'Primary key for the encounter';
COMMENT ON COLUMN public."encounters"."START" IS 'Time the encounter started';
COMMENT ON COLUMN public."encounters"."STOP" IS 'Time the encounter ended';
COMMENT ON COLUMN public."encounters"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."encounters"."ORGANIZATION" IS 'Organization providing care';
COMMENT ON COLUMN public."encounters"."PROVIDER" IS 'Primary provider for the encounter';
COMMENT ON COLUMN public."encounters"."PAYER" IS 'Payer responsible for the encounter';
COMMENT ON COLUMN public."encounters"."ENCOUNTERCLASS" IS 'Type of encounter';
COMMENT ON COLUMN public."encounters"."CODE" IS 'Encounter code';
COMMENT ON COLUMN public."encounters"."DESCRIPTION" IS 'Description of the encounter';
COMMENT ON COLUMN public."encounters"."BASE_ENCOUNTER_COST" IS 'Base cost of the encounter';
COMMENT ON COLUMN public."encounters"."TOTAL_CLAIM_COST" IS 'Total claim cost';
COMMENT ON COLUMN public."encounters"."PAYER_COVERAGE" IS 'Amount covered by the payer';
COMMENT ON COLUMN public."encounters"."REASONCODE" IS 'Reason code for the encounter';
COMMENT ON COLUMN public."encounters"."REASONDESCRIPTION" IS 'Reason description for the encounter';
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
COMMENT ON COLUMN public."imaging_studies"."Id" IS 'Primary key for the imaging study';
COMMENT ON COLUMN public."imaging_studies"."DATE" IS 'Date the imaging was performed';
COMMENT ON COLUMN public."imaging_studies"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."imaging_studies"."ENCOUNTER" IS 'Encounter related to the imaging';
COMMENT ON COLUMN public."imaging_studies"."BODYSITE_CODE" IS 'Code for body site imaged';
COMMENT ON COLUMN public."imaging_studies"."BODYSITE_DESCRIPTION" IS 'Description of the body site';
COMMENT ON COLUMN public."imaging_studies"."MODALITY_CODE" IS 'Imaging modality code';
COMMENT ON COLUMN public."imaging_studies"."MODALITY_DESCRIPTION" IS 'Imaging modality description';
COMMENT ON COLUMN public."imaging_studies"."SOP_CODE" IS 'SOP code for the image';
COMMENT ON COLUMN public."imaging_studies"."SOP_DESCRIPTION" IS 'SOP description for the image';
-- immunizations: vaccinations given to a patient
CREATE TABLE public.immunizations (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  BASE_COST numeric
);
COMMENT ON COLUMN public."immunizations"."DATE" IS 'Date the immunization was given';
COMMENT ON COLUMN public."immunizations"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."immunizations"."ENCOUNTER" IS 'Encounter during which the immunization occurred';
COMMENT ON COLUMN public."immunizations"."CODE" IS 'Immunization code';
COMMENT ON COLUMN public."immunizations"."DESCRIPTION" IS 'Description of the immunization';
COMMENT ON COLUMN public."immunizations"."BASE_COST" IS 'Base cost of the immunization';
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
COMMENT ON COLUMN public."medications"."START" IS 'Time medication was started';
COMMENT ON COLUMN public."medications"."STOP" IS 'Time medication was stopped';
COMMENT ON COLUMN public."medications"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."medications"."ENCOUNTER" IS 'Encounter related to the medication';
COMMENT ON COLUMN public."medications"."CODE" IS 'Medication code';
COMMENT ON COLUMN public."medications"."DESCRIPTION" IS 'Medication description';
COMMENT ON COLUMN public."medications"."BASE_COST" IS 'Base medication cost';
COMMENT ON COLUMN public."medications"."PAYER_COVERAGE" IS 'Amount covered by payer';
COMMENT ON COLUMN public."medications"."DISPENSES" IS 'Number of dispenses';
COMMENT ON COLUMN public."medications"."TOTALCOST" IS 'Total medication cost';
COMMENT ON COLUMN public."medications"."REASONCODE" IS 'Reason code for medication';
COMMENT ON COLUMN public."medications"."REASONDESCRIPTION" IS 'Reason description for medication';
COMMENT ON COLUMN public."medications"."PAYER" IS 'Payer responsible for medication';
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
COMMENT ON COLUMN public."observations"."DATE" IS 'Date of the observation';
COMMENT ON COLUMN public."observations"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."observations"."ENCOUNTER" IS 'Encounter associated with the observation';
COMMENT ON COLUMN public."observations"."CODE" IS 'Observation code';
COMMENT ON COLUMN public."observations"."DESCRIPTION" IS 'Observation description';
COMMENT ON COLUMN public."observations"."VALUE" IS 'Observed value';
COMMENT ON COLUMN public."observations"."UNITS" IS 'Units of the observed value';
COMMENT ON COLUMN public."observations"."TYPE" IS 'Type of observation';
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
COMMENT ON COLUMN public."organizations"."Id" IS 'Primary key for the organization';
COMMENT ON COLUMN public."organizations"."NAME" IS 'Organization name';
COMMENT ON COLUMN public."organizations"."ADDRESS" IS 'Street address of the organization';
COMMENT ON COLUMN public."organizations"."CITY" IS 'City where the organization is located';
COMMENT ON COLUMN public."organizations"."STATE" IS 'State of the organization';
COMMENT ON COLUMN public."organizations"."ZIP" IS 'Postal code';
COMMENT ON COLUMN public."organizations"."LAT" IS 'Latitude coordinate';
COMMENT ON COLUMN public."organizations"."LON" IS 'Longitude coordinate';
COMMENT ON COLUMN public."organizations"."PHONE" IS 'Contact phone number';
COMMENT ON COLUMN public."organizations"."REVENUE" IS 'Organization revenue';
COMMENT ON COLUMN public."organizations"."UTILIZATION" IS 'Utilization rate';
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
COMMENT ON COLUMN public."patients"."Id" IS 'Primary key for the patient';
COMMENT ON COLUMN public."patients"."BIRTHDATE" IS 'Date of birth';
COMMENT ON COLUMN public."patients"."DEATHDATE" IS 'Date of death';
COMMENT ON COLUMN public."patients"."SSN" IS 'Social security number';
COMMENT ON COLUMN public."patients"."DRIVERS" IS 'Driver license number';
COMMENT ON COLUMN public."patients"."PASSPORT" IS 'Passport number';
COMMENT ON COLUMN public."patients"."PREFIX" IS 'Name prefix';
COMMENT ON COLUMN public."patients"."FIRST" IS 'First name';
COMMENT ON COLUMN public."patients"."LAST" IS 'Last name';
COMMENT ON COLUMN public."patients"."SUFFIX" IS 'Name suffix';
COMMENT ON COLUMN public."patients"."MAIDEN" IS 'Maiden name';
COMMENT ON COLUMN public."patients"."MARITAL" IS 'Marital status';
COMMENT ON COLUMN public."patients"."RACE" IS 'Race of the patient';
COMMENT ON COLUMN public."patients"."ETHNICITY" IS 'Ethnicity of the patient';
COMMENT ON COLUMN public."patients"."GENDER" IS 'Gender of the patient';
COMMENT ON COLUMN public."patients"."BIRTHPLACE" IS 'Birthplace of the patient';
COMMENT ON COLUMN public."patients"."ADDRESS" IS 'Street address';
COMMENT ON COLUMN public."patients"."CITY" IS 'City of residence';
COMMENT ON COLUMN public."patients"."STATE" IS 'State of residence';
COMMENT ON COLUMN public."patients"."COUNTY" IS 'County of residence';
COMMENT ON COLUMN public."patients"."ZIP" IS 'Zip code';
COMMENT ON COLUMN public."patients"."LAT" IS 'Latitude coordinate';
COMMENT ON COLUMN public."patients"."LON" IS 'Longitude coordinate';
COMMENT ON COLUMN public."patients"."HEALTHCARE_EXPENSES" IS 'Total healthcare expenses';
COMMENT ON COLUMN public."patients"."HEALTHCARE_COVERAGE" IS 'Total healthcare coverage amount';
-- payer_transitions: yearly insurance coverage changes for a patient
CREATE TABLE public.payer_transitions (
  PATIENT uuid,
  START_YEAR integer,
  END_YEAR integer,
  PAYER uuid,
  OWNERSHIP text
);
COMMENT ON COLUMN public."payer_transitions"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."payer_transitions"."START_YEAR" IS 'Year coverage started';
COMMENT ON COLUMN public."payer_transitions"."END_YEAR" IS 'Year coverage ended';
COMMENT ON COLUMN public."payer_transitions"."PAYER" IS 'Payer during the coverage period';
COMMENT ON COLUMN public."payer_transitions"."OWNERSHIP" IS 'Ownership type of the policy';
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
COMMENT ON COLUMN public."payers"."Id" IS 'Primary key for the payer';
COMMENT ON COLUMN public."payers"."NAME" IS 'Payer name';
COMMENT ON COLUMN public."payers"."ADDRESS" IS 'Address of the payer';
COMMENT ON COLUMN public."payers"."CITY" IS 'City where the payer is located';
COMMENT ON COLUMN public."payers"."STATE_HEADQUARTERED" IS 'State where payer is headquartered';
COMMENT ON COLUMN public."payers"."ZIP" IS 'Zip code';
COMMENT ON COLUMN public."payers"."PHONE" IS 'Contact phone number';
COMMENT ON COLUMN public."payers"."AMOUNT_COVERED" IS 'Amount covered by the payer';
COMMENT ON COLUMN public."payers"."AMOUNT_UNCOVERED" IS 'Amount not covered';
COMMENT ON COLUMN public."payers"."REVENUE" IS 'Total revenue of the payer';
COMMENT ON COLUMN public."payers"."COVERED_ENCOUNTERS" IS 'Number of encounters covered';
COMMENT ON COLUMN public."payers"."UNCOVERED_ENCOUNTERS" IS 'Number of encounters not covered';
COMMENT ON COLUMN public."payers"."COVERED_MEDICATIONS" IS 'Number of medications covered';
COMMENT ON COLUMN public."payers"."UNCOVERED_MEDICATIONS" IS 'Number of medications not covered';
COMMENT ON COLUMN public."payers"."COVERED_PROCEDURES" IS 'Number of procedures covered';
COMMENT ON COLUMN public."payers"."UNCOVERED_PROCEDURES" IS 'Number of procedures not covered';
COMMENT ON COLUMN public."payers"."COVERED_IMMUNIZATIONS" IS 'Number of immunizations covered';
COMMENT ON COLUMN public."payers"."UNCOVERED_IMMUNIZATIONS" IS 'Number of immunizations not covered';
COMMENT ON COLUMN public."payers"."UNIQUE_CUSTOMERS" IS 'Count of unique customers';
COMMENT ON COLUMN public."payers"."QOLS_AVG" IS 'Average quality of life score';
COMMENT ON COLUMN public."payers"."MEMBER_MONTHS" IS 'Total member months';
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
COMMENT ON COLUMN public."procedures"."DATE" IS 'Date of the procedure';
COMMENT ON COLUMN public."procedures"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."procedures"."ENCOUNTER" IS 'Encounter associated with the procedure';
COMMENT ON COLUMN public."procedures"."CODE" IS 'Procedure code';
COMMENT ON COLUMN public."procedures"."DESCRIPTION" IS 'Procedure description';
COMMENT ON COLUMN public."procedures"."BASE_COST" IS 'Base cost of the procedure';
COMMENT ON COLUMN public."procedures"."REASONCODE" IS 'Reason code for the procedure';
COMMENT ON COLUMN public."procedures"."REASONDESCRIPTION" IS 'Reason description for the procedure';
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
COMMENT ON COLUMN public."providers"."Id" IS 'Primary key for the provider';
COMMENT ON COLUMN public."providers"."ORGANIZATION" IS 'Identifier of the organization';
COMMENT ON COLUMN public."providers"."NAME" IS 'Provider name';
COMMENT ON COLUMN public."providers"."GENDER" IS 'Gender of the provider';
COMMENT ON COLUMN public."providers"."SPECIALITY" IS 'Provider specialty';
COMMENT ON COLUMN public."providers"."ADDRESS" IS 'Street address';
COMMENT ON COLUMN public."providers"."CITY" IS 'City';
COMMENT ON COLUMN public."providers"."STATE" IS 'State';
COMMENT ON COLUMN public."providers"."ZIP" IS 'Zip code';
COMMENT ON COLUMN public."providers"."LAT" IS 'Latitude coordinate';
COMMENT ON COLUMN public."providers"."LON" IS 'Longitude coordinate';
COMMENT ON COLUMN public."providers"."UTILIZATION" IS 'Utilization metric';
-- supplies: non‑drug items dispensed to patients
CREATE TABLE public.supplies (
  DATE timestamp with time zone,
  PATIENT uuid,
  ENCOUNTER uuid,
  CODE integer,
  DESCRIPTION text,
  QUANTITY integer
);
COMMENT ON COLUMN public."supplies"."DATE" IS 'Date the supply was provided';
COMMENT ON COLUMN public."supplies"."PATIENT" IS 'Identifier of the patient';
COMMENT ON COLUMN public."supplies"."ENCOUNTER" IS 'Encounter related to the supply';
COMMENT ON COLUMN public."supplies"."CODE" IS 'Supply code';
COMMENT ON COLUMN public."supplies"."DESCRIPTION" IS 'Description of the supply';
COMMENT ON COLUMN public."supplies"."QUANTITY" IS 'Quantity dispensed';
