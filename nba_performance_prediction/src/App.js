import logo from "./logo.svg";
import "./App.css";
import { GiBasketballBall, GiMedal } from "react-icons/gi";
import { IoIosBasketball } from "react-icons/io";
import React from "react";

function ContainerHeader({ text, icon = null }) {
  return (
    <div className="h-20">
      <div className="h-1/4 rounded-t-3xl gradient bg-black mb-50 group-hover:h-full transition-all duration-100 ease-linear"></div>
      <div
        className=" h-3/4 flex gradient text-transparent bg-clip-text mx-10 items-end justify-left z-50 
      group-hover:text-white group-hover:items-center group-hover:mt-[-70px]
      transition-all duration-100 ease-linear"
      >
        <span className="flex h-10 items-center text-cyan-500 mr-3 group-hover:text-white group-hover:rotate-[720deg] transition-all ease-in-out duration-[400ms]">
          {icon}{" "}
        </span>
        <span className=" flex items-center h-10 text-xl font-extrabold text-shadow-sm">
          {text}
        </span>
      </div>
    </div>
  );
}

function FormField({
  outsidetext = null,
  icon = null,
  onchange = null,
  elementname = null,
  value = null
}) {
  return (
    <div className=" flex flex-col align-left w-1/10 mb-5">
      <span className=" gradient text-transparent bg-clip-text pl-2">
        {outsidetext}
      </span>
      <div className="flex flex-row w-full">
        {/* <div className="gradient mt-2 p-2 rounded-l-lg"/> */}
        <input
          className="mt-1 h-9 rounded-lg shadow-md border-transparent outline-none text-gray-500 text-xs focus:shadow-gray-400 shadow-gray-200 focus:appearance-none appearance-none w-full py-2 px-3 leading-tight focus:outline-none transition-all duration-75 ease-linear"
          type={"number"}
          placeholder={outsidetext}
          min="0"
          onChange={onchange}
          name={elementname}
          value = {value}
          step="0.01"
        />
      </div>
    </div>
  );
}

class Customform extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      GP: "36",
      PTS: "7.3",
      FGM: "7",
      "3PMade": "0.5",
      FTM: "1.6",
      OREB: "0.7",
      REB: "4.1",
      STL: "0.4",
      MIN: "28",
      FGA: "12",
      "3PA": "2",
      FTA: "2",
      DREB: "1.2",
      AST: "3",
      BLK: "1",
      TOV: "0.3",
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({ [event.target.name]: event.target.value });
    console.log(this.state);
  }

  getscore(score) {
    return 100 * score.toFixed(2);
  }

  changescoretext(respJ) {
    document.getElementById("scorecontainer").textContent =
      this.getscore(respJ.score) + "%";
    return respJ;
  }

  handleSubmit(event) {
    event.preventDefault();

    console.log(this.state);
    const url = "http://localhost:8000/scoreJson";
    const bodydata = JSON.stringify({
      GP: this.state.GP,
      PTS: this.state.PTS,
      FGM: this.state.FGM,
      "3PMade": this.state["3PMade"],
      FTM: this.state.FTM,
      OREB: this.state.OREB,
      REB: this.state.REB,
      STL: this.state.STL,
      MIN: this.state.MIN,
      FGA: this.state.FGA,
      "3PA": this.state["3PA"],
      FTA: this.state.FTA,
      DREB: this.state.DREB,
      AST: this.state.AST,
      BLK: this.state.BLK,
      TOV: this.state.TOV,
    });
    const reqOpt = {
      method: "POST",
      headers: { "Content-type": "application/json" },
      body: bodydata,
    };
    fetch(url, reqOpt)
      .then((resp) => resp.json())
      .then((respJ) => this.changescoretext(respJ))
      .then((respJ) => document.getElementById('resultext').textContent = this.gettext(respJ));
  }

  gettext(respJ) {
    if (respJ.prediction === 0) {
      return (
        "It's not looking good bruv... Looks like your player didn't make it past 5 years in the NBA. maybe consider sending him to boot camp? We estimate there's a " +
        this.getscore(respJ.score) +
        "%  chance he'll make it through"
      );
    } else {
      return (
        "Nice! Your player probably will make it through his years at the NBA. We estimate there's a " +
        this.getscore(respJ.score) +
        "%  chance he'll make it through"
      );
    }
  }

  render() {
    return (
      <form
        className="flex flex-col h-[75%] overflow-auto scrollbar mb-5 pb-5 mt-5"
        onSubmit={this.handleSubmit}
      >
        <div className="flex flex-row h-auto mt-5 mx-10 justify-around ">
          <div className="w-[45%] flex flex-col align-left">
            <FormField
              outsidetext={"Games played"}
              onchange={this.handleChange}
              elementname="GP"
              value = "36"
            />
            <FormField
              outsidetext={"Points / game "}
              onchange={this.handleChange}
              elementname="PTS"
              value = "7.3"
            />
            <FormField
              outsidetext={"Field goals made"}
              onchange={this.handleChange}
              elementname="FGM"
              value="7.0"
            />
            <FormField
              outsidetext={"3 points Made"}
              onchange={this.handleChange}
              elementname="3PMade"
              value="0.5"
            />
            <FormField
              outsidetext={"Free Throws Made"}
              onchange={this.handleChange}
              elementname="FTM"
              value="1.6"

            />
            <FormField
              outsidetext={"Offensive rebounds"}
              onchange={this.handleChange}
              elementname="OREB"
              value="0.7"
            />
            <FormField
              outsidetext={"Rebounds"}
              onchange={this.handleChange}
              elementname="REB"
              value = "4.1"
            />
            <FormField
              outsidetext={"Steals"}
              onchange={this.handleChange}
              elementname="STL"
              value="0.4"
            />
          </div>
          <div className="w-[45%] flex flex-col align-left">
            <FormField
              outsidetext={"Minutes played"}
              onchange={this.handleChange}
              elementname="MIN"
              value = "11"
            />
            <FormField
              outsidetext={"Field goal attempts"}
              onchange={this.handleChange}
              elementname="FGA"
              value = "12.0"
            />
            <FormField
              outsidetext={"3 Point attempts"}
              onchange={this.handleChange}
              elementname="3PA"
              value = "2"
            />
            <FormField
              outsidetext={"Free throw attemps"}
              onchange={this.handleChange}
              elementname="FTA"
              value = "2"
            />
            <FormField
              outsidetext={"Defensive rebounds"}
              onchange={this.handleChange}
              elementname="DREB"
              value = "1.2"
            />
            <FormField
              outsidetext={"Assists"}
              onchange={this.handleChange}
              elementname="AST"
              value = "3.0"
            />
            <FormField
              outsidetext={"Blocks"}
              onchange={this.handleChange}
              elementname="BLK"
              value = "1.0"
            />
            <FormField
              outsidetext={"Turnover"}
              onchange={this.handleChange}
              elementname="TOV"
              value = "0.3"
            />
          </div>
        </div>
        <div className="flex flex-row align-middle justify-center">
          <input
            type="submit"
            className="hover:cursor-pointer hover:shadow-lg gradient hover:bg-blue-200 mt-10 justify-center flex h-10 w-32 text-white font-extrabold shadow-md rounded-xl hover:scale-105 transition-all duration-75 ease-linear"
          />
        </div>
      </form>
    );
  }
}

function App() {
  return (
    <div className="flex flex-col lg:flex-row lg:p-30 h-screen lg:w-screen">
      <div className="flex flex-col container h-fill w-full lg:w-1/3 my-20 ml-20 mr-10 group ">
        <ContainerHeader
          text={"Choose your player!"}
          icon={<IoIosBasketball size="24" />}
        />
        <div className=" flex mt-5 text-black text-sm mx-10 h-auto items-start justify-left">
          <span>
            Input the parameters for your player and we'll tell you if he's a
            good fit for the NBA! Our model is trained on ~1200 lines of data
            with 10-fold cross validation and extensive grid searches to get the
            best recall.
          </span>
        </div>
        <Customform />
      </div>

      <div className="h-auto w-full lg:w-3/4 mx-10 my-20">
        <div className="container group h-auto lg:w-1/2 flex flex-col">
          <ContainerHeader
            text="Results"
            icon={<GiMedal size={"24"} />}
          ></ContainerHeader>
          <span
            id="scorecontainer"
            className=" flex flex-row grow mt-5 flex-1 w-full items-center align-middle justify-center font-mono font-extrabold text-3xl text-blue-400"
          ></span>
          <span id = 'resultext' className="text-sm mt-5 mb-5 mx-5"></span>
        </div>
        {/* <div className="container border-transparent border-2 col-span-2"></div>
        <div className="container border-transparent border-2 col-span-3"></div>
        <div className="container border-transparent border-2 col-span-2"></div> */}
      </div>
    </div>
  );
}

export default App;
