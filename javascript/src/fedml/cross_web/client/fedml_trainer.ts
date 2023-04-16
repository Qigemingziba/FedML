export class FedMLTrainer {
  trainer
  client_index
  trainData: any
  trainDataLabel: any
  testData: any
  testDataLabel: any
  local_sample_number: null
  train_local: null
  test_local: null
  device
  args

  constructor(
    client_index: any,
    trainData: any,
    trainDataLabel: any,
    testData: any,
    testDataLabel: any,
    device: any,
    args: any,
    model_trainer: any,
  ) {
    this.trainer = model_trainer
    this.client_index = client_index
    this.trainData = trainData
    this.trainDataLabel = trainDataLabel
    this.testData = testData
    this.testDataLabel = testDataLabel
    this.train_local = null
    this.local_sample_number = null
    this.test_local = null
    this.device = device
    this.args = args
  }

  async train(round_idx: number) {
    this.args.round_idx = round_idx
    await this.trainer.train()
    const weights = this.trainer.get_model_params()
    return weights
  }

  update_model(weights: any) {
    console.log('update_model fedml_trainer.ts ', weights)
    this.trainer.set_model_params(weights)
  }
}