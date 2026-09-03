const LOCAL_API_URL = "http://localhost:8000/api";
const PRODUCTION_API_URL = "https://name-tesis-lstm-gru-backend.onrender.com/api";

function resolveApiUrl() {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return LOCAL_API_URL;
    }
  }

  return PRODUCTION_API_URL;
}

const API_URL = resolveApiUrl();
export type DomainId = "phishing" | "energia" | "finanzas";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public path: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface DomainOption {
  id: DomainId;
  title: string;
  source: string;
  description: string;
}

export interface DashboardData {
  dataset: {
    filename: string;
    originalRows: number;
    cleanedRows: number;
    rowsRemoved: number;
    columns: string[];
    dataTypes: Record<string, string>;
  };
  typeDistribution: { name: string; value: number }[];
  columnBarData: { col: string; tipo: string; registros: number }[];
  numericDistribution: { rango: string; cantidad: number }[];
  numericColumn?: string | null;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ModelPerformance {
  f1: number;
  precision: number;
  recall: number;
  rmse?: number;
  confusionMatrix: ConfusionMatrix;
  detectedCount: number;
}

export interface EvaluatedData {
  filename: string;
  totalRows: number;
  realAnomaliesCount: number;
  models: {
    lstm: ModelPerformance;
    gru: ModelPerformance;
    brnn: ModelPerformance;
    transformer: ModelPerformance;
    tcn: ModelPerformance;
  };
  timeline: {
    date: string;
    actual: number;
    anomalies: number;
    lstm: number;
    gru: number;
    brnn: number;
    transformer: number;
    tcn: number;
  }[];
  samples: {
    id: number;
    label: string;
    value: number;
    real: string;
    lstm: string;
    gru: string;
    brnn: string;
    transformer: string;
    tcn: string;
  }[];
  processedRecords?: Record<string, unknown>[];
}

export interface ComparisonData {
  filename: string;
  radarData: Record<string, string | number>[];
  comparisonBarData: Record<string, string | number>[];
  scatterData: { model: string; time: number; accuracy: number }[];
  comparisonTable: {
    metric: string;
    lstm: number;
    gru: number;
    brnn: number;
    transformer: number;
    tcn: number;
    winner: string;
  }[];
}

export interface HistoryItem {
  id: number;
  date: string;
  domain: string;
  data: string;
  model: string;
  confidence: number;
  realLabel: string;
  predicted: string;
}

export interface HistoryData {
  filename: string;
  items: HistoryItem[];
}

export interface ExternalSource {
  id: string;
  name: string;
  provider: string;
  url: string;
  official: boolean;
  requiresKey: boolean;
  configured: boolean;
  useCase: string;
}

export interface ExternalSourceResult {
  source: ExternalSource;
  status: "ok" | "error" | "needs_key" | "configured" | "reference";
  count: number;
  records: Record<string, unknown>[];
  error?: string | null;
}

export interface ExternalData {
  domain: DomainId;
  updatedAt: string;
  results: ExternalSourceResult[];
}

export interface DataLakeDomainSummary {
  domain: DomainId;
  target: number;
  updatedAt: string | null;
  totalRecords: number;
  sourceBreakdown: Record<string, number>;
  categoryBreakdown: Record<string, number>;
}

export interface DataLakeSummary {
  updatedAt: string;
  totalRecords: number;
  domains: DataLakeDomainSummary[];
}

export interface DataLakeRecords {
  domain: DomainId;
  page: number;
  pageSize: number;
  totalRecords: number;
  totalPages: number;
  records: Record<string, unknown>[];
}

export interface BestModelSummary {
  model: string;
  score: number;
  f1: number;
  precision: number;
  recall: number;
  rmse: number;
  trainTime: number;
}

export interface DomainMetricsSummary {
  domain: DomainId;
  totalRows: number;
  realAnomaliesCount: number;
  bestModel: BestModelSummary;
  models: Record<string, BestModelSummary & { detectedCount: number }>;
}

export interface MetricsSummary {
  createdAt: string;
  domains: DomainMetricsSummary[];
  overallBest: (BestModelSummary & { domain: DomainId }) | null;
}

export interface TrainingManifest {
  runId: string;
  mode: string;
  limit: number;
  createdAt: string;
  domainTotals: Record<DomainId, number>;
  metricsSummary: MetricsSummary;
  models: { domain: DomainId; model: string; latestPath: string; experimentPath: string }[];
  paths: Record<string, string>;
}

export interface ExperimentListItem {
  runId: string;
  mode: string;
  limit: number;
  createdAt: string;
  domainTotals: Record<DomainId, number>;
  metricsSummary: MetricsSummary;
  path: string;
}

export interface ScientificDataDomainSummary {
  id: DomainId;
  label: string;
  unit: string;
  available: boolean;
  datasetId: string | null;
  source: string | null;
  originalRows: number;
  usableRows: number;
  developmentRows: number;
  trainingRows: number | null;
  validationRows: number;
  lockedTestRows: number;
  oofUniqueRows: number;
  testSetLocked: boolean;
  testSetUsed: boolean;
  summaryOrigin?: "live_manifest" | "versioned_snapshot" | "unavailable";
  classDistribution?: { negative: number; positive: number };
  range?: [string | null, string | null];
}

export interface ScientificDataSummary {
  schemaVersion: string;
  generatedAt: string;
  dataOrigin?: "live_manifests" | "versioned_snapshot" | "mixed" | "unavailable";
  snapshotId?: string | null;
  available: boolean;
  availableDomains: number;
  totalDomains: number;
  totalUsableObservations: number;
  domains: ScientificDataDomainSummary[];
  countingPolicy: string;
  demoExcluded: boolean;
  finalTestUsed: boolean;
}

export interface EnergyComparisonRow {
  predictorId: string;
  displayName: string;
  family: "base" | "ensemble";
  foldEvaluations: number;
  rmseMean: number;
  rmseStd: number;
  maeMean: number;
  smapeMean: number;
  r2Mean: number;
  trainTimeMean: number | null;
  inferenceTimeMsMean: number | null;
}

export interface EnergyExperimentView {
  available: true;
  run: {
    runId: string;
    status: "demo" | "thesis_candidate";
    createdAt: string;
    protocolVersion: string;
  };
  dataset: {
    rows: number;
    features: string[];
    target: string;
  };
  validation: {
    folds: Record<string, unknown>[];
    seeds: number[];
    window: number;
    horizon: number;
    testSetUsed: boolean;
    classificationMetricsAvailable: boolean;
    anomalyLabelType: string;
  };
  comparison: EnergyComparisonRow[];
  winner: {
    overall: string | null;
    overallDisplayName: string | null;
    bestEnsemble: string | null;
    bestEnsembleDisplayName: string | null;
    primaryMetric: "rmse";
    lowerIsBetter: true;
  };
  timeline: ({ timestamp: string; actual: number; ensemble: number } & Record<string, string | number>)[];
  anomalies: {
    recommendedMethod: string;
    labelType: string;
    classificationMetricsAvailable: boolean;
    summaries: {
      predictorFamily: string;
      predictorId: string;
      displayName: string;
      method: string;
      evaluatedSamples: number;
      estimatedAnomalies: number;
      estimatedAnomalyRate: number;
      meanScore: number;
      maxScore: number;
    }[];
  };
  methodology: {
    testSetUsed: boolean;
    testSetEvaluatedOnce: boolean;
    stackingReady: boolean;
    warning: string;
  };
}

export type EnergyExperimentResponse =
  | EnergyExperimentView
  | { available: false; message: string };

export interface EnergyOptimizedRevalidationView {
  available: true;
  runId: string;
  status: "thesis_optimized_candidate" | "demo_revalidation";
  protocol: {
    selectionFolds: number[];
    independentComparisonFold: number;
    selectionUsesIndependentFold: false;
    testSetUsed: false;
    bootstrapUnit: "calendar_day";
    bootstrapIterations: number;
  };
  selection: { selectedAblationId: string; selectedBaseModels: string[]; selectionMetric: "rmse" };
  independentComparison: {
    metrics: Array<{ predictorId: string; rmse: number; mae: number; smape: number; r2: number }>;
    ranking: string[];
    bestBaseModel: string;
    optimizedStackingIsWinner: boolean;
    pairedInference: {
      referenceId: string;
      candidateId: string;
      observedDeltaRmse: number;
      confidenceInterval95: [number, number];
      probabilityCandidateBetter: number;
      calendarDays: number;
      iterations: number;
    };
  };
  xai: { method: string; items: Array<{ feature: string; importanceMean: number; importanceStd: number }> };
}

export type EnergyOptimizedRevalidationResponse = EnergyOptimizedRevalidationView | { available: false; message: string };

export interface EnergyExperimentRequest {
  protocol?: "demo" | "thesis";
  source?: "sample" | "silver";
  window?: number;
  horizon?: number;
  gapSteps?: number;
  folds?: number;
  epochs?: number;
  batchSize?: number;
  seeds?: number[];
  modelIds?: ("lstm" | "gru" | "brnn" | "tcn" | "transformer")[];
}

export type EnergyJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";

export interface EnergyExperimentJob {
  jobId: string;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  config: Required<EnergyExperimentRequest>;
  progress: {
    stage: string;
    event: string;
    completedUnits: number;
    totalUnits: number;
    percent: number;
    seed: number | null;
    fold: number | null;
    modelId: string | null;
    message: string;
  };
  resultRunId: string | null;
  error: string | null;
}

export type EnergyJobResponse = EnergyExperimentJob | { available: false; message: string };

export interface EnergyDatasetStatus {
  available: boolean;
  readyForThesisPilot: boolean;
  message?: string;
  datasetId?: string;
  provider?: string;
  sourceVersion?: string;
  retrievedAt?: string;
  preparedAt?: string;
  silver?: { path: string; sha256: string; bytes: number; rows: number; columns: string[] };
  selectedSegment?: { startAt: string; endAt: string; rows: number; yearsApprox: number; targetMissingValues: number; featureMissingValues: Record<string, number> };
  readiness?: { minimumRows: number; ready: boolean; reasons: string[]; targetImputed: boolean; featuresImputedGlobally: boolean; testSetUsed: boolean };
  sourceAudit?: Record<string, unknown>;
}

export interface FinanceMarketDataStatus {
  available: boolean;
  readyForThesisTraining: boolean;
  message: string;
  datasetId?: string;
  provider?: string;
  category?: string;
  datasetSlug?: string;
  resourceId?: string;
  sourceUrl?: string;
  retrievedAt?: string;
  preparedAt?: string;
  integrityVerified?: boolean;
  silver?: { path: string; sha256: string; bytes: number; rows: number; columns: string[] };
  scientificContract?: {
    task: string;
    unitOfAnalysis: string;
    target: string;
    targetDefinition: string;
    forecastHorizon: string;
    frequency: string;
    groundTruthAnomalyLabels: boolean;
    prohibitedClaim: string;
  };
  audit?: {
    source: {
      originalRows: number;
      usableRows: number;
      startAt: string;
      endAt: string;
      duplicateTimestamps: number;
      conflictingDuplicateDates: number;
      targetDuplicateConflicts: number;
      realIndexZeroPlaceholdersConvertedToMissing: number;
    };
    target: { column: string; definition: string; unit: string; minimum: number; maximum: number; mean: number; standardDeviation: number };
    chronologicalSplit: {
      developmentRows: number;
      lockedTestRows: number;
      developmentRange: [string, string];
      lockedTestRange: [string, string];
      testSetLocked: boolean;
      testSetUsed: boolean;
      testEvaluationAuthorized: boolean;
    };
    readiness: { ready: boolean; reasons: string[]; groundTruthAnomalyLabelsAvailable: boolean; fraudClaimsAllowed: boolean; testSetUsed: boolean };
  };
  testPolicy: { testSetLocked: boolean; testSetUsed: boolean; testEvaluationAuthorized: boolean };
}

export interface FinanceMarketFreezeReadiness {
  ready: boolean;
  totalChecks: number;
  passedChecks: number;
  blockingCheckIds: string[];
  testSetUsed: boolean;
  testEvaluationAuthorized: boolean;
}

export interface FinanceMarketFreeze {
  available?: boolean;
  freezeId?: string;
  status?: string;
  createdAt?: string;
  testPolicy?: { testSetLocked: boolean; testSetUsed: boolean; testEvaluationAuthorized: boolean };
  message?: string;
}

export interface FinanceDatasetStatus {
  available: boolean;
  datasetId?: string;
  task: string;
  provider?: string;
  message: string;
  lineageCurrent?: boolean;
  readyForPipelinePilot: boolean;
  readyForThesisTraining: boolean;
  citation?: { title: string; authors: string; year: number; url: string; repository: string; repositoryCommit: string };
  provenance?: { kind: string; upstreamDataFilesExecuted: boolean; upstreamPicklesLoaded: boolean; generatorPolicy: string; methodologyLicense: string };
  contract?: {
    eventId: string;
    eventTime: string;
    entityId: string;
    counterpartyId: string;
    amount: string;
    target: string;
    targetMeaning: Record<string, string>;
    columns: string[];
  };
  configuration?: { days: number; customers: number; terminals: number; seed: number; startDate: string; splitPolicy: string };
  silver?: { path: string; sha256: string; bytes: number; rows: number; columns: string[] };
  audit?: {
    rows: number;
    customers: number;
    terminals: number;
    fraudRows: number;
    fraudRate: number;
    splits: Array<{ split: string; rows: number; fraudRows: number; fraudRate: number; startAt: string; endAt: string; labels: number[] }>;
    temporalAudit: { chronological: boolean; testLocked: boolean; testEvaluated: boolean; futureInformationUsed: boolean };
    readiness: { readyForPipelinePilot: boolean; readyForThesisTraining: boolean; reasons: string[]; thesisReasons: string[] };
  };
  testLock: { locked: boolean; evaluated: boolean; maximumEvaluations?: number; policy?: string };
}

export interface FinanceBenchmarkRequest {
  days?: number;
  customers?: number;
  terminals?: number;
  seed?: number;
}

export interface FinanceRealDataStatus {
  available: boolean;
  templateAvailable: boolean;
  readyForPreparation: boolean;
  readyForThesisTraining: boolean;
  active?: boolean;
  activeDatasetId?: string | null;
  adapter?: "ieee_cis" | "ulb_worldline";
  packageId?: string;
  manifestPath: string;
  message: string;
  source?: {
    sourceId?: string;
    provider?: string;
    sourceUrl?: string;
    sourceVersion?: string;
    retrievedAt?: string;
    datasetKind?: string;
    realWorld?: boolean;
    labelsAreProviderGroundTruth?: boolean;
  };
  license?: { name?: string; url?: string; accepted?: boolean; acceptedAt?: string | null; acceptedBy?: string };
  checks?: {
    passed: boolean;
    reasons: string[];
    adapterSupported?: boolean;
    realWorldDeclared?: boolean;
    providerLabelsDeclared?: boolean;
    licenseAccepted?: boolean;
    sourceVersionFixed?: boolean;
    filesVerified?: boolean;
    sequenceEntityAvailable?: boolean;
    providerTestUsed?: boolean;
    targetUsedAsFeature?: boolean;
    futureInformationAllowed?: boolean;
  };
  catalog?: {
    selection: { recommendedAdapter: string; reason: string; automaticDownload: boolean; testSetPolicy: string };
    items: Array<{
      sourceId: string;
      displayName: string;
      provider: string;
      sourceUrl: string;
      licenseName: string;
      adapter: string;
      requiredFile: string;
      supportsSequenceEntity: boolean;
      thesisRole: string;
    }>;
  };
}

export interface FinanceRealPreparationRequest {
  minimumRows?: number;
  minimumFraudPerSplit?: number;
  minimumEntities?: number;
}

export interface FinanceSequenceStatus {
  available: boolean;
  datasetId?: string;
  createdAt?: string;
  message: string;
  lineageCurrent?: boolean;
  artifactIntegrity?: boolean;
  readyForBaseModelPilot: boolean;
  readyForThesisTraining: boolean;
  configuration?: {
    window: number;
    folds: number;
    purgeDays: number;
    padding: string;
    sequenceOrder: string;
  };
  features?: {
    columns: string[];
    sequenceColumns: string[];
    target: string;
    targetUsedAsFeature: boolean;
    historicalLabelsUsedAsFeatures: boolean;
    causalPolicy: string;
    validationOnlinePolicy: string;
  };
  sequences?: {
    rows: number;
    developmentRows: number;
    trainingRows: number;
    externalValidationRows: number;
    testRowsAvailable: number;
    testRowsEncoded: number;
    shape: number[];
  };
  oof?: {
    strategy: string;
    coverageRows: number;
    warmupRowsExcluded: number;
    assignmentUniquenessPassed: boolean;
    futureRowsUsedForFit: boolean;
    scalerPolicy: string;
    folds: Array<{
      fold: number;
      fitRows: number;
      holdoutRows: number;
      fitStartAt: string;
      fitEndAt: string;
      holdoutStartAt: string;
      holdoutEndAt: string;
      purgeDays: number;
      fitFraudRows: number;
      holdoutFraudRows: number;
      futureRowsUsedForFit: boolean;
      scaler: { fitPolicy: string; fitRows: number; featureColumns: string[]; mean: number[]; scale: number[] };
    }>;
  };
  testLock: { locked: boolean; evaluated: boolean; encoded: boolean; rows?: number; policy?: string };
}

export interface FinanceSequenceRequest {
  window?: number;
  folds?: number;
  purgeDays?: number;
}

export interface FinanceExperimentRequest {
  protocol: "demo" | "thesis";
  epochs: number;
  batchSize: number;
  patience: number;
  seeds: number[];
  modelIds: Array<"lstm" | "gru" | "brnn" | "tcn" | "transformer">;
  demoMaxRowsPerFold: number | null;
}

export interface FinanceExperimentJob {
  jobId: string;
  executionRunId: string;
  resumedFromJobId: string | null;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  config: FinanceExperimentRequest;
  progress: {
    stage: string;
    event: string;
    completedUnits: number;
    resumedUnits: number;
    totalUnits: number;
    percent: number;
    seed: number | null;
    fold: number | null;
    modelId: string | null;
    message: string;
  };
  resultRunId: string | null;
  error: string | null;
}

export interface FinanceExperimentView {
  available: true;
  run: { runId: string; status: string; createdAt: string; protocol: string };
  configuration: FinanceExperimentRequest & { primaryMetric: string; threshold: number; thresholdPolicy: string };
  execution: { resumable: boolean; resumeCount: number; resumedUnits: number; trainedUnitsThisInvocation: number };
  dataset: { datasetId: string; oofRowsAvailablePerSeed: number; oofRowsUsedPerSeed: number; demoSubset: boolean };
  validation: {
    futureLeakagePassed: boolean;
    externalValidationUsed: boolean;
    externalValidationReserved: boolean;
    testSetLocked: boolean;
    testSetEncoded: boolean;
    testSetUsed: boolean;
  };
  comparison: Array<{
    modelId: string;
    displayName: string;
    foldEvaluations: number;
    prAucMean: number;
    prAucStd: number;
    rocAucMean: number;
    f1Mean: number;
    precisionMean: number;
    recallMean: number;
    mccMean: number;
    balancedAccuracyMean: number;
    brierScoreMean: number;
    logLossMean: number;
    falsePositiveRateMean: number;
    trainTimeSecondsMean: number;
    inferenceTimeMsPerSampleMean: number;
    modelSizeBytesMean: number;
  }>;
  winner: { modelId: string | null; displayName: string | null; primaryMetric: string; higherIsBetter: boolean };
  stacking: { ready: boolean; featureColumns: string[]; trained: boolean };
  methodology: { testSetLocked: boolean; testSetUsed: boolean; externalValidationUsed: boolean; warning: string };
}

export type FinanceExperimentResponse = FinanceExperimentView | { available: false; message: string };
export type FinanceJobResponse = FinanceExperimentJob | { available: false; message: string };

export interface FinanceStackingComparisonRow {
  candidateId: string;
  sourceCandidateId?: string;
  family: "base" | "baseline" | "stacking";
  foldEvaluations: number;
  prAucMean: number;
  prAucStd: number;
  rocAucMean: number;
  f1Mean: number;
  precisionMean: number;
  recallMean: number;
  mccMean: number;
  balancedAccuracyMean: number;
  brierScoreMean: number;
  logLossMean: number;
  falsePositiveRateMean: number;
}

export interface FinanceStackingView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  baseRun: { runId: string; rows: number; models: string[]; oofSha256: string };
  metaFeatures: { columns: string[]; baseProbabilityColumns: string[]; summaryFeatures: string[]; targetIncluded: boolean };
  validation: {
    strategy: string;
    evaluatedFolds: number[];
    coverageRows: number;
    warmupRowsExcluded: number;
    sameRowsForAllCandidates: boolean;
    directTransactionOverlapPassed: boolean;
    strictTemporalCrossFit: boolean;
    metaDependencyLeakageControlled: boolean;
    futureFoldsUsed: boolean;
    externalValidationUsed: boolean;
    testSetLocked: boolean;
    testSetEncoded: boolean;
    testSetUsed: boolean;
    interpretation: string;
  };
  candidateAggregate: FinanceStackingComparisonRow[];
  baseAggregateOnCommonRows: FinanceStackingComparisonRow[];
  comparison: FinanceStackingComparisonRow[];
  sixModelComparison: FinanceStackingComparisonRow[];
  recommendation: {
    leadingStackingCandidateId: string | null;
    baseLeaderId: string | null;
    overallSixModelLeaderId: string | null;
    stackingBeatsBestBaseOnMeanPrAuc: boolean;
    primaryMetric: string;
    status: string;
  };
  thresholdPolicy: { current: number; status: string; finalSelectionData: string; testSetUsed: boolean };
}

export type FinanceStackingResponse = FinanceStackingView | { available: false; message: string };

export interface FinanceValidationRequest {
  demoMaxTrainRows: number | null;
  demoMaxValidationRows: number | null;
  bootstrapIterations: number;
}

export interface FinanceValidationJob {
  jobId: string;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  config: FinanceValidationRequest;
  progress: {
    stage: string;
    event: string;
    completedUnits: number;
    resumedUnits: number;
    totalUnits: number;
    percent: number;
    seed: number | null;
    modelId: string | null;
    message: string;
  };
  resultRunId: string | null;
  error: string | null;
}

export interface FinanceValidationComparisonRow {
  candidateId: string;
  sourceCandidateId?: string | null;
  family: "base" | "stacking";
  validationSelectionRows: number;
  calibrationMethod: "identity" | "platt" | "isotonic";
  calibratedThreshold: number;
  thresholdObjective: string;
  prAuc: number;
  rocAuc: number;
  f1: number;
  precision: number;
  recall: number;
  mcc: number;
  balancedAccuracy: number;
  brierScore: number;
  logLoss: number;
  falsePositiveRate: number;
}

export interface FinanceValidationView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  dataset: {
    kind: string;
    trainRowsAvailable: number;
    trainRowsUsed: number;
    validationRowsAvailable: number;
    validationRowsUsed: number;
    demoTrainSubset: boolean;
    demoValidationSubset: boolean;
  };
  training: { resumable: boolean; resumedUnits: number; trainedUnitsThisInvocation: number };
  calibration: { strategy: string; rows: number; thresholdObjective: string; selectionLabelsUsed: boolean; testLabelsUsed: boolean };
  selection: {
    strategy: string;
    rows: number;
    winnerCandidateId: string;
    winnerThreshold: number;
    winnerCalibrationMethod: string;
    stackingRank: number;
    stackingBeatsBestBase: boolean;
    status: string;
  };
  comparison: FinanceValidationComparisonRow[];
  statisticalComparison: {
    stackingVersusBestBase: {
      bestBaseCandidateId: string;
      iterations: number;
      observedDelta: number;
      ciLower: number;
      ciUpper: number;
      probabilityStackingBetter: number;
      statisticallyClearAt95Percent: boolean;
    };
  };
  validation: {
    sameRowsForAllSixCandidates: boolean;
    chronologicalCalibrationBeforeSelection: boolean;
    futureLeakagePassed: boolean;
    externalRealDatasetUsed: boolean;
    testSetLocked: boolean;
    testSetEncoded: boolean;
    testSetUsed: boolean;
    readyForFinalTestEvaluation: boolean;
    interpretation: string;
  };
  freeze: {
    candidateId: string;
    calibrationMethod: string;
    threshold: number;
    syntheticBenchmark: boolean;
    testSetLocked: boolean;
    testSetUsed: boolean;
    eligibleForFinalThesisClaim: boolean;
  };
}

export type FinanceValidationResponse = FinanceValidationView | { available: false; message: string };
export type FinanceValidationJobResponse = FinanceValidationJob | { available: false; message: string };

export interface FinanceDiversityPair {
  modelA: string;
  modelB: string;
  probabilityPearson: number | null;
  probabilitySpearman: number | null;
  residualPearson: number | null;
  hardPredictionDisagreementRate: number;
  doubleFaultRate: number;
  bothCorrectRate: number;
  falsePositiveJaccard: number | null;
  falseNegativeJaccard: number | null;
  onlyModelACorrect: number;
  onlyModelBCorrect: number;
}

export interface FinanceAblationMetric {
  ablationId: string;
  removedModelId: string | null;
  retainedModels: string[];
  metaModelId: string;
  calibrationRows: number;
  selectionRows: number;
  calibrationMethod: "identity" | "platt" | "isotonic";
  calibratedThreshold: number;
  prAuc: number;
  rocAuc: number;
  f1: number;
  precision: number;
  recall: number;
  mcc: number;
  balancedAccuracy: number;
  brierScore: number;
  logLoss: number;
  falsePositiveRate: number;
  prAucDropVsFull: number;
  mccDropVsFull: number;
  f1DropVsFull: number;
  falsePositiveRateIncreaseVsFull: number;
  seedCount: number;
}

export interface FinanceModelContribution {
  modelId: string;
  referenceMetaModelId: string;
  prAucDropWhenRemoved: number;
  prAucDropCi95: { lower: number; upper: number };
  probabilityPositiveContribution: number;
  mccDropWhenRemoved: number;
  f1DropWhenRemoved: number;
  falsePositiveRateIncreaseWhenRemoved: number;
  interpretation: "positive_contribution" | "potentially_redundant_or_harmful";
}

export interface FinanceDiversityView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  validationRun: {
    runId: string;
    calibrationRows: number;
    selectionRows: number;
    datasetKind: string;
  };
  diversity: {
    rows: number;
    models: string[];
    thresholdSource: string;
    probabilitySource: string;
    pairs: FinanceDiversityPair[];
    meanPairwiseDisagreementRate: number;
    mostComplementaryPair: FinanceDiversityPair | null;
    mostCorrelatedPair: FinanceDiversityPair | null;
  };
  ablation: {
    strategy: string;
    metaModels: string[];
    configurations: Array<{ ablationId: string; removedModelId: string | null; retainedModels: string[] }>;
    metrics: FinanceAblationMetric[];
    referenceMetaModelId: string;
    contribution: FinanceModelContribution[];
  };
  stability: {
    seedCount: number;
    seedLevelInferenceAvailable: boolean;
    bootstrapUnit: string;
    bootstrapIterations: number;
    confidenceLevel: number;
    interpretation: string;
  };
  recommendation: {
    referenceMetaModelId: string;
    recommendedAblationId: string;
    recommendedBaseModels: string[];
    removedModelId: string | null;
    fullPrAuc: number;
    selectedPrAuc: number;
    ablationAccepted: boolean;
    configurationLeader: { ablationId: string; metaModelId: string; prAuc: number };
    contributionRanking: string[];
    status: string;
  };
  validation: {
    sameRowsForAllAblations: boolean;
    metaFitUsesOofOnly: boolean;
    calibrationUsesFirstValidationHalfOnly: boolean;
    selectionLabelsUsedForCalibration: boolean;
    selectionLabelsUsedForThreshold: boolean;
    calibrationBeforeSelection: boolean;
    testSetLocked: boolean;
    testSetEncoded: boolean;
    testSetUsed: boolean;
  };
}

export type FinanceDiversityResponse = FinanceDiversityView | { available: false; message: string };

export interface FinanceOptimizedRevalidationView {
  available: true;
  runId: string;
  status: string;
  protocol: { calibrationRows: number; ablationSelectionRows: number; independentComparisonRows: number; sameRowsForAllCandidates: boolean };
  stackingSelection: { selectedAblationId: string; selectedMetaModelId: string; selectedThreshold: number; configurationsCompared: number };
  independentComparison: {
    winnerCandidateId: string;
    stackingRank: number;
    stackingBeatsBestBase: boolean;
    metrics: Array<FinanceValidationComparisonRow & { ablationId: string | null; comparisonRows: number }>;
    stackingVersusBestBase: { bestBaseCandidateId: string; observedDelta: number; ciLower: number; ciUpper: number; probabilityStackingBetter: number; iterations: number };
  };
  validation: { testSetLocked: boolean; testSetEncoded: boolean; testSetUsed: boolean; eligibleForFreeze: boolean; interpretation: string };
}

export type FinanceOptimizedRevalidationResponse = FinanceOptimizedRevalidationView | { available: false; message: string };

export interface FinanceFreezeCheck {
  id: string;
  passed: boolean;
  expected: unknown;
  observed: unknown;
}

export interface FinanceFreezeReadiness {
  ready: boolean;
  status: "ready_for_freeze_creation" | "blocked";
  summary: { passed: number; failed: number; total: number };
  checks: FinanceFreezeCheck[];
  blockingCheckIds: string[];
  artifactVerificationRequested: boolean;
  lineage: Record<string, string>;
}

export interface FinanceFrozenPipelineView {
  available: true;
  reused: boolean;
  freezeId: string;
  status: "ready_for_single_final_test_evaluation";
  immutable: boolean;
  sourceFingerprint: string;
  sealVerified: boolean;
  gate: { passed: boolean; artifactCount: number; dependenciesVerifiedAtFreeze: boolean };
  configuration: {
    datasetId: string;
    datasetKind: string;
    baseModels: string[];
    seeds: number[];
    oofFolds: number[];
    selectedCandidateId: string;
    selectedStackingMetaModelId: string;
    stackingAblationId: string;
    calibratedThresholds: Record<string, number>;
    calibrationMethods: Record<string, string>;
    finalTestPolicy: string;
  };
  testAuthorization: { required: boolean; granted: boolean; evaluated: boolean; maximumEvaluations: number; policy: string };
}

export type FinanceFrozenPipelineResponse = FinanceFrozenPipelineView | { available: false; message: string };

export interface ThesisProtocolJob {
  jobId: string;
  executionRunId: string;
  resumedFromJobId: string | null;
  status: "queued" | "running" | "waiting_for_external_data" | "waiting_for_scientific_data" | "completed" | "failed" | "cancelled" | "interrupted" | "paused";
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  pauseRequested?: boolean;
  config: {
    epochs: number;
    batchSize: number;
    patience?: number;
    seeds: number[];
    bootstrapIterations: number;
    window?: number;
    horizon?: number;
    gapSteps?: number;
    folds?: number;
  };
  progress: { stage: string; percent: number; remainingPercent?: number; message: string };
  stages: Array<{ stage: string; status: string; completedAt: string; message: string }>;
  result: Record<string, string> | null;
  blocker: { code: string; message: string } | null;
  error: string | null;
  testPolicy: { testSetUsed: boolean; testEvaluationAuthorized: boolean };
}

export type ThesisProtocolJobResponse = ThesisProtocolJob | { available: false; message: string };

export interface ThesisExecutionDomainStatus {
  id: DomainId;
  label: string;
  route: "/phishing" | "/energy" | "/finance";
  dependsOn: DomainId[];
  status: "pending" | ThesisProtocolJob["status"] | "unknown";
  jobId: string | null;
  executionRunId: string | null;
  updatedAt: string | null;
  progress: { stage: string; percent: number; remainingPercent?: number; message: string };
  blocker: { code: string; message: string; manifestPath?: string } | null;
  nextAction: string;
  testPolicy: { testSetUsed: boolean; testEvaluationAuthorized: boolean };
}

export interface ThesisExecutionStatus {
  schemaVersion: string;
  generatedAt: string;
  scope: "three_domain_scientific_execution";
  progressMethod: {
    formula: string;
    equalDomainWeight: boolean;
    domainCount: number;
    includesSoftwareImplementation: false;
    note: string;
  };
  overall: {
    percent: number;
    remainingPercent?: number;
    completedDomains: number;
    totalDomains: number;
    status: string;
    activeDomain: DomainId | null;
  };
  domains: ThesisExecutionDomainStatus[];
  blockers: Array<{ domain: DomainId; code: string; message: string }>;
  chain: {
    status: string;
    domain: DomainId | null;
    message: string;
    updatedAt: string | null;
    ownerPid: number | null;
  };
  testPolicy: {
    locked: boolean;
    testSetUsed: boolean;
    testEvaluationAuthorized: boolean;
    maximumEvaluationsPerFrozenDomain: number;
    message: string;
  };
}

export interface ThesisResultCandidate {
  candidateId: "lstm" | "gru" | "brnn" | "tcn" | "transformer" | "stacking";
  sourceCandidateId: string;
  family: "base" | "stacking";
  primaryValue: number;
  secondary: Record<string, number | null>;
  rank: number;
}

export interface ThesisSeedDistribution {
  candidateId: ThesisResultCandidate["candidateId"];
  seeds: number[];
  values: number[];
  seedCount: number;
  minimum: number;
  q1: number;
  median: number;
  q3: number;
  maximum: number;
}

export interface ThesisResultDomainSummary {
  id: DomainId;
  label: string;
  available: boolean;
  taskType: "classification" | "regression";
  primaryMetric: "prAuc" | "rmse";
  higherIsBetter: boolean;
  comparisonSplit: string | null;
  runId: string | null;
  evidenceStatus: "unavailable" | "demonstration_only" | "scientific_candidate_pending_freeze" | "scientific_frozen_pre_test";
  eligibleForThesisConclusion: boolean;
  freezeSealValid: boolean;
  sixCandidateContractComplete: boolean;
  sixCandidateWinner: string | null;
  allEvaluatedCandidatesWinner: string | null;
  stackingRank: number | null;
  stackingBeatsBestBase: boolean | null;
  bestBaseCandidateId: string | null;
  comparison: ThesisResultCandidate[];
  seedDistributions: ThesisSeedDistribution[];
  pairedInference: Record<string, string | number | boolean> | null;
  testSetUsed: boolean;
  interpretation: string;
}

export interface ThesisResultsSummary {
  schemaVersion: string;
  generatedAt: string;
  title: string;
  candidateContract: {
    baseModels: string[];
    metaModelLabel: "stacking";
    expectedCandidatesPerDomain: 6;
    selectionSeparatedFromFinalTest: true;
  };
  domains: ThesisResultDomainSummary[];
  readiness: {
    eligibleDomains: number;
    totalDomains: number;
    allDomainsFrozen: boolean;
    finalTestUsed: boolean;
  };
  interpretationPolicy: string[];
}

export interface EnergyDataPreparationJob {
  jobId: string;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  force: boolean;
  progress: { stage: string; percent: number; message: string };
  result: { datasetId: string; rows: number; readyForThesisPilot: boolean } | null;
  error: string | null;
}

export type EnergyDataJobResponse = EnergyDataPreparationJob | { available: false; message: string };

export interface PhishingDatasetStatus {
  available: boolean;
  integrityVerified?: boolean;
  readyForPipelinePilot: boolean;
  readyForThesisTraining: boolean;
  message?: string;
  datasetId?: string;
  preparedAt?: string;
  providers?: string[];
  trancoPermanentListUrl?: string;
  silver?: { path: string; sha256: string; bytes: number; rows: number; columns: string[] };
  classDistribution?: { negative: number; positive: number; total: number; positiveRate: number };
  splitDistribution?: Record<string, { rows: number; negative: number; positive: number; groups: number }>;
  leakageAudit?: { groupKey: string; overlapCounts: Record<string, number>; passed: boolean };
  biasAudit?: {
    sourceLabelCoupling: boolean;
    labelsPerSource: Record<string, number>;
    sourcesPerLabel: Record<string, string[]>;
    mixedLabelSources: string[];
    negativeLabelsIndependentlyVerified: boolean;
    verifiedNegativeRows: number;
    negativeRows: number;
    positiveLabelsIndependentlyVerified: boolean;
    verifiedPositiveRows: number;
    positiveRows: number;
    urlShape: { rootPathRateByLabel: Record<string, number>; absoluteRootPathRateGap: number; maximumAcceptedGap: number };
  };
  readiness?: {
    readyForPipelinePilot: boolean;
    readyForThesisTraining: boolean;
    pilotReasons: string[];
    thesisReasons: string[];
    testSetUsed: boolean;
  };
}

export interface PhishingCurationStatus {
  available: boolean;
  valid: boolean;
  manifestPath: string;
  message?: string;
  studyId?: string;
  rows?: number;
  labels?: { negative: number; positive: number };
  sources?: Array<{
    sourceId: string;
    provider: string;
    citation: string;
    license: string;
    independentAcquisition: boolean;
    rows: number;
    labels: number[];
    negativeRows: number;
    verifiedNegativeRows: number;
    positiveRows: number;
    verifiedPositiveRows: number;
  }>;
  sourcesPerLabel?: Record<string, string[]>;
  mixedLabelSources?: string[];
  negativeEvidence?: { verified: number; total: number };
  positiveEvidence?: { verified: number; total: number };
  readyForScientificMerge?: boolean;
  scientificReasons?: string[];
  requirements: string[];
}

export interface PhishingCurationTemplateResult {
  created: boolean;
  directory: string;
  exampleManifestPath: string;
  exampleCsvPath: string;
  activation: string;
}

export interface PhishingDataPreparationJob {
  jobId: string;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  force: boolean;
  perClass: number;
  includeAcademicSources?: boolean;
  progress: { stage: string; percent: number; message: string };
  result: { datasetId: string; rows: number; readyForPipelinePilot: boolean; readyForThesisTraining: boolean; academicCurationReady?: boolean } | null;
  error: string | null;
}

export type PhishingDataJobResponse = PhishingDataPreparationJob | { available: false; message: string };

export interface PhishingSequenceStatus {
  available: boolean;
  integrityVerified?: boolean;
  readyForBaseModelTraining: boolean;
  lineageCurrent?: boolean;
  datasetScientificReady?: boolean;
  message?: string;
  datasetId?: string;
  createdAt?: string;
  configuration?: {
    folds: number;
    seed: number;
    maxVocabulary: number;
    lengthPercentile: number;
    maxLengthCap: number;
  };
  outerSplit?: {
    strategy: string;
    groupKey: string;
    distribution: Record<string, { rows: number; negative: number; positive: number; groups: number }>;
    groupOverlapCounts: Record<string, number>;
    passed: boolean;
  };
  testLock?: {
    locked: boolean;
    rows: number;
    groups: number;
    usedForVocabulary: boolean;
    usedForLengthSelection: boolean;
    usedForOOF: boolean;
    usedForThresholdSelection: boolean;
    evaluated: boolean;
  };
  oof?: {
    strategy: string;
    folds: number;
    seed: number;
    coverageRows: number;
    expectedRows: number;
    groupLeakagePassed: boolean;
    allHoldoutsContainBothClasses: boolean;
    foldsAudit: Array<{
      fold: number;
      fitRows: number;
      holdoutRows: number;
      fitGroups: number;
      holdoutGroups: number;
      holdoutNegative: number;
      holdoutPositive: number;
      groupOverlap: number;
      tokenizer: { vocabularySize: number; maxLength: number };
    }>;
  };
  tokenization?: {
    type: string;
    vocabularyFitSplit: string;
    maxLengthFitSplit: string;
    outerTokenizer: { vocabularySize: number; maxLength: number };
    validationStatistics: { unknownCharacterRate: number; truncatedRowRate: number };
    testStatisticsNotComputed: boolean;
  };
  readiness?: { readyForBaseModelTraining: boolean; reasons: string[]; testSetUsed: boolean };
}

export interface PhishingExperimentRequest {
  protocol: "demo" | "thesis";
  epochs: number;
  batchSize: number;
  patience: number;
  seeds: number[];
  modelIds: Array<"lstm" | "gru" | "brnn" | "tcn" | "transformer">;
  demoMaxRows: number | null;
}

export interface PhishingExperimentJob {
  jobId: string;
  executionRunId: string;
  resumedFromJobId: string | null;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  config: PhishingExperimentRequest;
  progress: {
    stage: string;
    event: string;
    completedUnits: number;
    resumedUnits: number;
    totalUnits: number;
    percent: number;
    seed: number | null;
    fold: number | null;
    modelId: string | null;
    message: string;
  };
  resultRunId: string | null;
  preflight: PhishingRuntimePreflight | null;
  error: string | null;
}

export interface PhishingRuntimePreflight {
  schemaVersion: string;
  checkedAt: string;
  protocol: "demo" | "thesis";
  ready: boolean;
  checks: Array<{ id: string; passed: boolean; message: string }>;
  warnings: string[];
  runtime: {
    platform: string;
    pythonVersion: string;
    tensorflowVersion: string;
    cpuCount: number | null;
    gpuCount: number;
    gpus: string[];
    executable: string;
  };
  storage: { outputPath: string; freeBytes: number; minimumFreeBytes: number };
  testPolicy: { locked: boolean; usedByPreflight: boolean; message: string };
}

export interface PhishingExperimentView {
  available: true;
  run: { runId: string; status: string; createdAt: string; protocol: string };
  configuration: PhishingExperimentRequest & { threshold: number; thresholdPolicy: string };
  dataset: { outerTrainRowsAvailable: number; rowsUsed: number; demoSubset: boolean };
  validation: { oofCoverageRowsPerSeed: number; groupLeakagePassed: boolean; testSetLocked: boolean; testSetUsed: boolean; outerValidationUsed: boolean };
  comparison: Array<{
    modelId: string;
    displayName: string;
    foldEvaluations: number;
    prAucMean: number;
    prAucStd: number;
    rocAucMean: number;
    f1Mean: number;
    precisionMean: number;
    recallMean: number;
    mccMean: number;
    falsePositiveRateMean: number;
    trainTimeSecondsMean: number;
    inferenceTimeMsPerSampleMean: number;
    modelSizeBytesMean: number;
  }>;
  winner: { modelId: string | null; displayName: string | null; primaryMetric: string; higherIsBetter: boolean };
  stacking: { ready: boolean; featureColumns: string[]; trained: boolean };
  methodology: { testSetLocked: boolean; testSetUsed: boolean; outerValidationUsed: boolean; thresholdStatus: string; warning: string };
}

export type PhishingExperimentResponse = PhishingExperimentView | { available: false; message: string };
export type PhishingJobResponse = PhishingExperimentJob | { available: false; message: string };

export interface PhishingStackingView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  baseRun: { runId: string; rows: number; models: string[] };
  metaFeatures: { columns: string[]; baseProbabilityColumns: string[]; summaryFeatures: string[] };
  validation: {
    strategy: string;
    coverageRows: number;
    directSampleOverlapPassed: boolean;
    strictNestedCrossFit: boolean;
    metaDependencyLeakageControlled: boolean;
    outerValidationUsed: boolean;
    testSetLocked: boolean;
    testSetUsed: boolean;
    interpretation: string;
  };
  comparison: Array<{
    candidateId: string;
    family: "base" | "baseline" | "stacking";
    foldEvaluations: number;
    prAucMean: number;
    prAucStd: number;
    rocAucMean: number;
    f1Mean: number;
    precisionMean: number;
    recallMean: number;
    mccMean: number;
    balancedAccuracyMean: number;
    falsePositiveRateMean: number;
  }>;
  recommendation: {
    leadingStackingCandidateId: string | null;
    overallOofLeaderId: string | null;
    primaryMetric: string;
    status: string;
  };
  thresholdPolicy: { current: number; status: string; finalSelectionData: string; testSetUsed: boolean };
}

export type PhishingStackingResponse = PhishingStackingView | { available: false; message: string };

export interface PhishingExternalValidationJob {
  jobId: string;
  status: EnergyJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  cancelRequested: boolean;
  progress: {
    stage: string;
    event: string;
    completedUnits: number;
    totalUnits: number;
    percent: number;
    seed: number | null;
    modelId: string | null;
    message: string;
  };
  resultRunId: string | null;
  error: string | null;
}

export interface PhishingExternalValidationView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  dataset: { developmentRows: number; validationRows: number; demoSubset: boolean; developmentValidationDomainOverlap: number };
  selection: {
    candidatePrimaryMetric: string;
    thresholdObjective: string;
    winnerCandidateId: string;
    winnerThreshold: number;
    leadingStackingCandidateId: string | null;
    status: string;
  };
  comparison: Array<{
    candidateId: string;
    family: "base" | "baseline" | "stacking";
    validationRows: number;
    seeds: number;
    calibratedThreshold: number;
    thresholdObjective: string;
    prAuc: number;
    rocAuc: number;
    f1: number;
    precision: number;
    recall: number;
    mcc: number;
    balancedAccuracy: number;
    falsePositiveRate: number;
    confusionMatrix: { trueNegative: number; falsePositive: number; falseNegative: number; truePositive: number };
  }>;
  validation: {
    outerValidationUsed: boolean;
    validationRows: number;
    sameRowsForAllCandidates: boolean;
    seedProbabilitiesAveragedForSelection: boolean;
    testSetLocked: boolean;
    testFeaturesEncoded: boolean;
    testSetUsed: boolean;
    interpretation: string;
  };
}

export type PhishingExternalValidationResponse = PhishingExternalValidationView | { available: false; message: string };
export type PhishingExternalValidationJobResponse = PhishingExternalValidationJob | { available: false; message: string };

export interface PhishingDiversityAblationView {
  available: true;
  runId: string;
  status: string;
  createdAt: string;
  diversity: {
    rows: number;
    models: string[];
    meanPairwiseDisagreementRate: number;
    mostComplementaryPair: PhishingDiversityPair | null;
    mostCorrelatedPair: PhishingDiversityPair | null;
    pairs: PhishingDiversityPair[];
  };
  ablation: {
    strategy: string;
    metaModels: string[];
    referenceMetaModelId: string;
    metrics: PhishingAblationMetric[];
    contribution: PhishingModelContribution[];
  };
  stability: { seedCount: number; seedLevelInferenceAvailable: boolean; bootstrapUnit: string; bootstrapIterations: number; confidenceLevel: number };
  recommendation: {
    referenceMetaModelId: string;
    recommendedAblationId: string;
    recommendedBaseModels: string[];
    removedModelId: string | null;
    fullPrAuc: number;
    selectedPrAuc: number;
    configurationLeader: { ablationId: string; metaModelId: string; prAuc: number };
    contributionRanking: string[];
    status: string;
  };
  validation: { outerValidationUsed: boolean; sameRowsForAllAblations: boolean; testSetLocked: boolean; testFeaturesEncoded: boolean; testSetUsed: boolean };
}

export interface PhishingDiversityPair {
  modelA: string;
  modelB: string;
  probabilityPearson: number | null;
  probabilitySpearman: number | null;
  residualPearson: number | null;
  hardPredictionDisagreementRate: number;
  doubleFaultRate: number;
  falsePositiveJaccard: number | null;
  falseNegativeJaccard: number | null;
  onlyModelACorrect: number;
  onlyModelBCorrect: number;
}

export interface PhishingAblationMetric {
  ablationId: string;
  removedModelId: string | null;
  retainedModels: string[];
  metaModelId: string;
  validationRows: number;
  calibratedThreshold: number;
  prAuc: number;
  f1: number;
  mcc: number;
  recall: number;
  falsePositiveRate: number;
  prAucDropVsFull: number;
  mccDropVsFull: number;
  f1DropVsFull: number;
}

export interface PhishingModelContribution {
  modelId: string;
  referenceMetaModelId: string;
  prAucDropWhenRemoved: number;
  prAucDropCi95: { lower: number; upper: number };
  mccDropWhenRemoved: number;
  f1DropWhenRemoved: number;
  falsePositiveRateIncreaseWhenRemoved: number;
  interpretation: string;
}

export type PhishingDiversityAblationResponse = PhishingDiversityAblationView | { available: false; message: string };

export interface PhishingFreezeReadiness {
  ready: boolean;
  structurallyReady: boolean;
  artifactIntegrityVerified: boolean;
  integrityVerificationRequiredAtFreeze: boolean;
  status: "ready_for_freeze_creation" | "blocked";
  checks: Array<{ checkId: string; passed: boolean; expected: unknown; observed: unknown }>;
  failedChecks: Array<{ checkId: string; passed: false; expected: unknown; observed: unknown }>;
  reasons: string[];
  lineage: Record<string, string>;
  requirements: {
    minimumSeeds: number;
    minimumFolds: number;
    requiredModels: string[];
    completeDevelopmentRequired: boolean;
    datasetScientificReadinessRequired: boolean;
    testMustRemainLocked: boolean;
  };
}

export interface PhishingFrozenPipelineView {
  available: true;
  reused: boolean;
  freezeId: string;
  status: "ready_for_single_test_evaluation";
  immutable: boolean;
  sourceFingerprint: string;
  sealVerified: boolean;
  gate: { passed: boolean; artifactCount: number; dependenciesVerifiedAtFreeze: boolean };
  configuration: {
    baseModels: string[];
    allComparedBaseModels: string[];
    seeds: number[];
    oofFolds: number[];
    stackingCandidateId: string;
    validationPrimaryCandidateId: string;
    frozenCandidateIds: string[];
    calibratedThresholds: Record<string, number>;
    finalTestPolicy: string;
  };
  testAuthorization: { required: boolean; granted: boolean; evaluated: boolean; maximumEvaluations: number; policy: string };
}

export type PhishingFrozenPipelineResponse = PhishingFrozenPipelineView | { available: false; message: string };

function withDomain(path: string, domain?: DomainId) {
  if (!domain) return path;
  const query = new URLSearchParams({ domain });
  return `${path}?${query.toString()}`;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new ApiError(payload?.detail || `Backend respondio ${response.status} en ${path}`, response.status, path);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new ApiError(payload?.detail || `Backend respondio ${response.status} en ${path}`, response.status, path);
  }
  return response.json() as Promise<T>;
}

export function fetchDomains() {
  return fetchJson<{ items: DomainOption[] }>("/domains");
}

export function fetchThesisExecutionStatus() {
  return fetchJson<ThesisExecutionStatus>("/thesis/status");
}

export function fetchThesisResultsSummary() {
  return fetchJson<ThesisResultsSummary>("/thesis/results-summary");
}

export function fetchDashboardData(domain?: DomainId) {
  return fetchJson<DashboardData>(withDomain("/dashboard", domain));
}

export function fetchAnalysisData(domain?: DomainId) {
  return fetchJson<EvaluatedData>(withDomain("/analysis", domain));
}

export function fetchComparisonData(domain?: DomainId) {
  return fetchJson<ComparisonData>(withDomain("/comparison", domain));
}

export function fetchHistoryData(domain?: DomainId) {
  return fetchJson<HistoryData>(withDomain("/history", domain));
}

export function fetchXaiData(domain?: DomainId) {
  return fetchJson<unknown>(withDomain("/xai", domain));
}

export async function fetchExternalData(domain: DomainId, limit = 5000) {
  const query = new URLSearchParams({ domain, limit: String(limit) });
  try {
    return await fetchJson<ExternalData>(`/external-data?${query.toString()}`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 422 && limit > 100) {
      const fallbackQuery = new URLSearchParams({ domain, limit: "100" });
      return fetchJson<ExternalData>(`/external-data?${fallbackQuery.toString()}`);
    }
    throw error;
  }
}

export function fetchDataLakeSummary() {
  return fetchJson<DataLakeSummary>("/data-lake/summary");
}

export function fetchDataLakeRecords(domain: DomainId, page = 1, pageSize = 100) {
  const query = new URLSearchParams({ domain, page: String(page), pageSize: String(pageSize) });
  return fetchJson<DataLakeRecords>(`/data-lake/records?${query.toString()}`);
}

export async function ingestDataLake(domain: DomainId | "all" = "all", target = 5000) {
  const query = new URLSearchParams({ domain, target: String(target) });
  const response = await fetch(`${API_URL}/data-lake/ingest?${query.toString()}`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Backend respondio ${response.status} al actualizar data lake`);
  }
  return response.json() as Promise<DataLakeSummary>;
}

export function fetchMetricsSummary() {
  return fetchJson<MetricsSummary>("/metrics-summary");
}

export function fetchTrainingManifest() {
  return fetchJson<TrainingManifest>("/training-manifest");
}

export function fetchExperiments() {
  return fetchJson<{ items: ExperimentListItem[] }>("/experiments");
}

export function fetchScientificDataSummary() {
  return fetchJson<ScientificDataSummary>("/scientific-data-summary");
}

export function fetchLatestEnergyExperiment() {
  return fetchJson<EnergyExperimentResponse>("/energy/experiments/latest");
}

export function runEnergyExperiment(request: EnergyExperimentRequest = {}) {
  return postJson<EnergyExperimentJob>("/energy/experiments/run", request);
}

export function fetchLatestEnergyJob() {
  return fetchJson<EnergyJobResponse>("/energy/jobs/latest");
}

export function fetchEnergyJob(jobId: string) {
  return fetchJson<EnergyExperimentJob>(`/energy/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelEnergyJob(jobId: string) {
  return postJson<EnergyExperimentJob>(`/energy/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function fetchLatestEnergyThesisProtocol() {
  return fetchJson<ThesisProtocolJobResponse>("/energy/thesis/latest");
}

export function runEnergyThesisProtocol() {
  return postJson<ThesisProtocolJob>("/energy/thesis/run", {
    window: 24,
    horizon: 1,
    gapSteps: 24,
    folds: 5,
    epochs: 20,
    batchSize: 32,
    seeds: [42, 101, 202, 303, 404],
    bootstrapIterations: 500,
  });
}

export function resumeEnergyThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/energy/thesis/${encodeURIComponent(jobId)}/resume`, {});
}

export function cancelEnergyThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/energy/thesis/${encodeURIComponent(jobId)}/cancel`, {});
}

export function pauseEnergyThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/energy/thesis/${encodeURIComponent(jobId)}/pause`, {});
}

export function fetchLatestEnergyOptimizedRevalidation() {
  return fetchJson<EnergyOptimizedRevalidationResponse>("/energy/revalidation/latest");
}

export function fetchEnergyDatasetStatus() {
  return fetchJson<EnergyDatasetStatus>("/energy/data");
}

export function fetchFinanceDatasetStatus() {
  return fetchJson<FinanceDatasetStatus>("/finance/data");
}

export function fetchFinanceMarketDataStatus() {
  return fetchJson<FinanceMarketDataStatus>("/finance/market-data");
}

export function prepareFinanceMarketData(force = false) {
  return postJson<FinanceMarketDataStatus>("/finance/market-data/prepare", { force });
}

export function fetchLatestFinanceMarketExperiment() {
  return fetchJson<EnergyExperimentResponse>("/finance/market-experiments/latest");
}

export function fetchLatestFinanceMarketRevalidation() {
  return fetchJson<EnergyOptimizedRevalidationResponse>("/finance/market-revalidation/latest");
}

export function fetchFinanceMarketFreezeReadiness() {
  return fetchJson<FinanceMarketFreezeReadiness>("/finance/market-freeze/readiness");
}

export function fetchLatestFinanceMarketFreeze() {
  return fetchJson<FinanceMarketFreeze>("/finance/market-freeze/latest");
}

export function createFinanceMarketFreeze() {
  return postJson<FinanceMarketFreeze>("/finance/market-freeze/create", {});
}

export function prepareFinanceBenchmark(request: FinanceBenchmarkRequest = {}) {
  return postJson<FinanceDatasetStatus>("/finance/data/prepare", {
    days: request.days ?? 100,
    customers: request.customers ?? 500,
    terminals: request.terminals ?? 200,
    seed: request.seed ?? 42,
  });
}

export function fetchFinanceRealDataStatus() {
  return fetchJson<FinanceRealDataStatus>("/finance/real-data");
}

export function initializeFinanceRealDataTemplate(adapter: "ieee_cis" | "ulb_worldline" = "ieee_cis", force = false) {
  return postJson<FinanceRealDataStatus>("/finance/real-data/template", { adapter, force });
}

export function prepareFinanceRealData(request: FinanceRealPreparationRequest = {}) {
  return postJson<{ realData: FinanceRealDataStatus; dataset: FinanceDatasetStatus }>("/finance/real-data/prepare", {
    minimumRows: request.minimumRows ?? 100_000,
    minimumFraudPerSplit: request.minimumFraudPerSplit ?? 100,
    minimumEntities: request.minimumEntities ?? 1_000,
  });
}

export function fetchFinanceSequenceStatus() {
  return fetchJson<FinanceSequenceStatus>("/finance/sequences");
}

export function prepareFinanceSequences(request: FinanceSequenceRequest = {}) {
  return postJson<FinanceSequenceStatus>("/finance/sequences/prepare", {
    window: request.window ?? 10,
    folds: request.folds ?? 5,
    purgeDays: request.purgeDays ?? 1,
  });
}

export function fetchLatestFinanceExperiment() {
  return fetchJson<FinanceExperimentResponse>("/finance/experiments/latest");
}

export function runFinanceExperiment(request: FinanceExperimentRequest) {
  return postJson<FinanceExperimentJob>("/finance/experiments/run", request);
}

export function fetchLatestFinanceJob() {
  return fetchJson<FinanceJobResponse>("/finance/jobs/latest");
}

export function fetchFinanceJob(jobId: string) {
  return fetchJson<FinanceExperimentJob>(`/finance/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelFinanceJob(jobId: string) {
  return postJson<FinanceExperimentJob>(`/finance/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function resumeFinanceJob(jobId: string) {
  return postJson<FinanceExperimentJob>(`/finance/jobs/${encodeURIComponent(jobId)}/resume`, {});
}

export function fetchLatestFinanceStacking() {
  return fetchJson<FinanceStackingResponse>("/finance/stacking/latest");
}

export function runFinanceStacking() {
  return postJson<FinanceStackingView>("/finance/stacking/run", {});
}

export function fetchLatestFinanceValidation() {
  return fetchJson<FinanceValidationResponse>("/finance/validation/latest");
}

export function runFinanceValidation(request: FinanceValidationRequest) {
  return postJson<FinanceValidationJob>("/finance/validation/run", request);
}

export function fetchLatestFinanceValidationJob() {
  return fetchJson<FinanceValidationJobResponse>("/finance/validation/jobs/latest");
}

export function cancelFinanceValidationJob(jobId: string) {
  return postJson<FinanceValidationJob>(`/finance/validation/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function fetchLatestFinanceDiversity() {
  return fetchJson<FinanceDiversityResponse>("/finance/diversity/latest");
}

export function runFinanceDiversity(bootstrapIterations = 300) {
  return postJson<FinanceDiversityView>("/finance/diversity/run", { bootstrapIterations });
}

export function fetchLatestFinanceOptimizedRevalidation() {
  return fetchJson<FinanceOptimizedRevalidationResponse>("/finance/revalidation/latest");
}

export function runFinanceOptimizedRevalidation(bootstrapIterations = 500) {
  return postJson<FinanceOptimizedRevalidationView>("/finance/revalidation/run", { bootstrapIterations });
}

export function fetchFinanceFreezeReadiness() {
  return fetchJson<FinanceFreezeReadiness>("/finance/freeze/readiness");
}

export function fetchLatestFinanceFreeze() {
  return fetchJson<FinanceFrozenPipelineResponse>("/finance/freeze/latest");
}

export function createFinanceFreeze() {
  return postJson<FinanceFrozenPipelineView>("/finance/freeze/create", {});
}

export function fetchLatestFinanceThesisProtocol() {
  return fetchJson<ThesisProtocolJobResponse>("/finance/thesis/latest");
}

export function runFinanceThesisProtocol() {
  return postJson<ThesisProtocolJob>("/finance/thesis/run", {
    window: 20,
    horizon: 1,
    gapSteps: 5,
    folds: 5,
    epochs: 20,
    batchSize: 32,
    seeds: [42, 101, 202, 303, 404],
    bootstrapIterations: 500,
  });
}

export function resumeFinanceThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/finance/thesis/${encodeURIComponent(jobId)}/resume`, {});
}

export function cancelFinanceThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/finance/thesis/${encodeURIComponent(jobId)}/cancel`, {});
}

export function pauseFinanceThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/finance/thesis/${encodeURIComponent(jobId)}/pause`, {});
}

export function fetchLatestPhishingThesisProtocol() {
  return fetchJson<ThesisProtocolJobResponse>("/phishing/thesis/latest");
}

export function runPhishingThesisProtocol() {
  return postJson<ThesisProtocolJob>("/phishing/thesis/run", { epochs: 20, batchSize: 64, patience: 4, seeds: [42, 101, 202, 303, 404], bootstrapIterations: 500 });
}

export function resumePhishingThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/phishing/thesis/${encodeURIComponent(jobId)}/resume`, {});
}

export function cancelPhishingThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/phishing/thesis/${encodeURIComponent(jobId)}/cancel`, {});
}

export function pausePhishingThesisProtocol(jobId: string) {
  return postJson<ThesisProtocolJob>(`/phishing/thesis/${encodeURIComponent(jobId)}/pause`, {});
}

export function fetchLatestEnergyDataJob() {
  return fetchJson<EnergyDataJobResponse>("/energy/data/jobs/latest");
}

export function prepareEnergyDataset(force = false) {
  return postJson<EnergyDataPreparationJob>("/energy/data/prepare", { force });
}

export function fetchPhishingDatasetStatus() {
  return fetchJson<PhishingDatasetStatus>("/phishing/data");
}

export function fetchLatestPhishingDataJob() {
  return fetchJson<PhishingDataJobResponse>("/phishing/data/jobs/latest");
}

export function preparePhishingDataset(force = false, perClass = 10_000, includeAcademicSources = false) {
  return postJson<PhishingDataPreparationJob>("/phishing/data/prepare", { force, perClass, includeAcademicSources });
}

export function fetchPhishingCurationStatus() {
  return fetchJson<PhishingCurationStatus>("/phishing/data/curation");
}

export function initializePhishingCurationTemplate() {
  return postJson<PhishingCurationTemplateResult>("/phishing/data/curation/template", {});
}

export function fetchPhishingSequenceStatus() {
  return fetchJson<PhishingSequenceStatus>("/phishing/sequences");
}

export function preparePhishingSequences() {
  return postJson<{ manifest: unknown; audit: unknown }>("/phishing/sequences/prepare", {
    folds: 5,
    seed: 42,
    maxVocabulary: 256,
    lengthPercentile: 99,
    maxLengthCap: 512,
  });
}

export function fetchLatestPhishingExperiment() {
  return fetchJson<PhishingExperimentResponse>("/phishing/experiments/latest");
}

export function runPhishingExperiment(request: PhishingExperimentRequest) {
  return postJson<PhishingExperimentJob>("/phishing/experiments/run", request);
}

export function fetchLatestPhishingJob() {
  return fetchJson<PhishingJobResponse>("/phishing/jobs/latest");
}

export function fetchPhishingRuntimePreflight(protocol: "demo" | "thesis" = "thesis") {
  return fetchJson<PhishingRuntimePreflight>(`/phishing/runtime/preflight?protocol=${encodeURIComponent(protocol)}`);
}

export function fetchPhishingJob(jobId: string) {
  return fetchJson<PhishingExperimentJob>(`/phishing/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelPhishingJob(jobId: string) {
  return postJson<PhishingExperimentJob>(`/phishing/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function resumePhishingJob(jobId: string) {
  return postJson<PhishingExperimentJob>(`/phishing/jobs/${encodeURIComponent(jobId)}/resume`, {});
}

export function fetchLatestPhishingStacking() {
  return fetchJson<PhishingStackingResponse>("/phishing/stacking/latest");
}

export function runPhishingStacking() {
  return postJson<PhishingStackingView>("/phishing/stacking/run", {});
}

export function fetchLatestPhishingExternalValidation() {
  return fetchJson<PhishingExternalValidationResponse>("/phishing/validation/latest");
}

export function runPhishingExternalValidation() {
  return postJson<PhishingExternalValidationJob>("/phishing/validation/run", {});
}

export function fetchLatestPhishingValidationJob() {
  return fetchJson<PhishingExternalValidationJobResponse>("/phishing/validation/jobs/latest");
}

export function fetchPhishingValidationJob(jobId: string) {
  return fetchJson<PhishingExternalValidationJob>(`/phishing/validation/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelPhishingValidationJob(jobId: string) {
  return postJson<PhishingExternalValidationJob>(`/phishing/validation/jobs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function fetchLatestPhishingDiversityAblation() {
  return fetchJson<PhishingDiversityAblationResponse>("/phishing/diversity/latest");
}

export function runPhishingDiversityAblation() {
  return postJson<PhishingDiversityAblationView>("/phishing/diversity/run", {});
}

export function fetchPhishingFreezeReadiness() {
  return fetchJson<PhishingFreezeReadiness>("/phishing/freeze/readiness");
}

export function fetchLatestPhishingFreeze() {
  return fetchJson<PhishingFrozenPipelineResponse>("/phishing/freeze/latest");
}

export function createPhishingFreeze() {
  return postJson<PhishingFrozenPipelineView>("/phishing/freeze/create", {});
}

export async function fetchAiAnalysis(type: "general" | "phishtank" | "energia" | "finanzas") {
  const query = new URLSearchParams({ type });
  const data = await fetchJson<{ analysis: string }>(`/ai-analysis?${query.toString()}`);
  return data.analysis;
}
