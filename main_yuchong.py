# The torch model I use is torch 1.8.1
import argparse
import matplotlib.pyplot as plt
# These are the lib for Transformer
from Transformermodel import TransAm
from Postional import PositionalEncoding
from Subfunctions import *

# These are the lib for LSTM
from Subfunction_LSTM import *

# Transformer model
# The input are the model, the date you want to do predict into the future, the hour you choose for the date and device.
# Since we train the model in cuda, so the device is automatically set into cuda
def main(
    load_trans_model = "96-48.pkl",
    load_lstm_model = "epoch=12-step=649.ckpt",
    path_to_save_prediction = "D:/pythonProject/AMI/combine/save_result/",
    choose_date = '2021-08-22',
    choose_hour = 23,
    Real_future = True,
    choose_future_day = '2021-08-29',# must be at least 7 days ahead the choose_date
    devices = "cuda"
):
    # This is the transformer part
    input_window = 96
    output_window = 48
    if devices == 'cpu':
        model_transformer = torch.load(load_trans_model,map_location='cpu')
    elif devices == 'cuda':
        model_transformer = torch.load(load_trans_model)
    model_transformer.eval()
    data = getDataFromAPI_HourlyIntervals('2021-05-10',choose_date)

    devices = torch.device(devices)
    input_data, scaler = get_data(data,choose_hour,input_window,output_window,devices)

    # This is only for testing the output of transformer
    # end_date = '2021-7-28'
    # data_real = getDataFromAPI_HourlyIntervals('2021-07-10',end_date)
    # value_real = data_real['Value'].values
    # value_real = scaler.fit_transform(value_real.reshape(-1, 1)).reshape(-1)
    # value_choose = value_real[-(input_window+output_window*steps+output_window+1):]

    steps = 4
    data = input_data[-1:]
    data = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    with torch.no_grad():
        for i in range(0, steps):
            output = model_transformer(data[-input_window:])
            # data_before = data
            data = torch.cat((data, output[-output_window:]))

    data = data.cpu().view(-1)
    data = scaler.inverse_transform(data.unsqueeze(1))
    trans_predict_7days = data[-192:-24]

  # # This is the LSTM part
    if choose_hour < 10:
        choose_hour = '0'+str(choose_hour)
    else:
        choose_hour = str(choose_hour)

    RequestedDatetime = pd.to_datetime(str(choose_date +' ' + choose_hour +':00'), format='%Y-%m-%d %H:00')
    pathToCheckpoint = load_lstm_model
    predictedDF = predict_price_LSTM(RequestedDatetime, pathToCheckpoint, historicalDays=180)
    predictedDF = predictedDF.set_index('Date')

    # Get real Values To compare with predicted
    if Real_future:
        MontelReal = getDataFromAPI_HourlyIntervals('2021-05-10', choose_future_day)
        MontelReal = MontelReal.drop(columns={'Base', 'Peak'})
        MontelReal = MontelReal.rename(columns={'Value': 'RealValue'})

        RealAndPred = pd.concat([MontelReal, predictedDF], axis=1)
        RealAndPred = RealAndPred.dropna()
        Real_value = RealAndPred['RealValue'].values
        Real_value = Real_value.reshape(-1, 1)
        if Real_value.shape == (168,1):
            Real_future = True
        else:
            Real_future = False

    LSTM_predicted_7days = predictedDF
    LSTM_predicted_7days = LSTM_predicted_7days['Value'].values
    LSTM_predicted_7days = LSTM_predicted_7days.reshape(-1,1)
    combine_value = 0.5*LSTM_predicted_7days + 0.5*trans_predict_7days

    # Real_value = RealAndPred['RealValue'].values
    # Real_value = Real_value.reshape(-1,1)

    print("The value one hour ahead {}".format(combine_value[0]))
    print("The value one day ahead {}".format(combine_value[23]))
    print("The value one week ahead {}".format(combine_value[-1]))

    plt.plot(combine_value,color='red',label='predict')
    if Real_future:
        plt.plot(Real_value,color = 'blue',label='real')
        plt.plot(Real_value-combine_value,color='green',label='different')
    figname = "predictionresult1212.jpg"
    plt.savefig(path_to_save_prediction+figname)
    # plt.legend()
    # plt.show()
    return combine_value

    return combine_value
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_trans_model",type=str,default="96-48.pkl")
    parser.add_argument("--load_lstm_model", type=str, default="epoch=12-step=649.ckpt")
    parser.add_argument("--path_to_save_prediction",type=str,default="D:/pythonProject/AMI/combine/save_result/")
    parser.add_argument("--choose_date", type=str, default='2021-07-10')
    parser.add_argument("--choose_hour", type=int, default=0)
    parser.add_argument("--Real_future", type=bool, default=True)
    parser.add_argument("--choose_future_day", type=str, default='2021-08-29')# must be at least 7 days ahead the choose_date
    parser.add_argument("--devices", type=str, default="cpu")
    args = parser.parse_args()

    main(
        load_trans_model=args.load_trans_model,
        load_lstm_model= args.load_lstm_model,
        path_to_save_prediction=args.path_to_save_prediction,
        choose_date = args.choose_date,
        choose_hour = args.choose_hour,
        Real_future= args.Real_future,
        choose_future_day= args.choose_future_day,
        devices =args.devices,
    )
