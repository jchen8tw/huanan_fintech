import React, { Component } from "react";
import {
  Form,
  Button,
  Container,
  Row,
  Col,
  Input,
  FormGroup,
  Label,
  Table
} from "reactstrap";

class Texta extends Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <Input
        onChange={this.props.onChange}
        type="textarea"
        name="get_result"
        id="exampleText"
      />
    );
  }
}
class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      text: "",
      result: null
    };
  }
  setText = e => {
    this.setState({ text: e.target.value });
  };
  subm = () => {
    //console.log(this.state.text)
    //console.log(JSON.stringify({payload: this.state.text}));
    fetch("/get_result", {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify({ data: this.state.text })
    })
      .then(res => res.json())
      .then(res => {
        this.setState({ result: res });
      });
  };
  render() {
    return (
      <Container>
        <Row style={{ textAlign: "center" }}>
          <Col>
            <h1>房貸預測</h1>
          </Col>
        </Row>
        <Row>
          <Col sm="4" />
          <Col sm="4" style={{ textAlign: "center" }}>
            <Form>
              <FormGroup>
                <Label for="exampleText">Input Condition</Label>
                <Texta onChange={this.setText} />
              </FormGroup>
              <Button onClick={this.subm} color="primary" size="lg">
                Submit
              </Button>
            </Form>
          </Col>
          <Col sm="4" />
        </Row>
        <Row>
          {this.state.result && <Table>
            <thead>
              <tr>
                <th>項目</th>
                <th>預測值</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(this.state.result).map(key => (
                <tr>
                  <th scope="row">{key}</th>
                  <td>{this.state.result[key]}</td>
                </tr>
              ))}
            </tbody>
          </Table>}
        </Row>
      </Container>
    );
  }
}

export default App;
