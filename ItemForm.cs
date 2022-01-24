using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;

namespace TAO_Enhancer
{
    public partial class ItemForm : Form
    {
        string identifier = "";
        //TODO 4: Tato pole možná nemusí být globální
        List<string> subquestionArray = new List<string>();
        List<string> possibleAnswerArray = new List<string>();
        List<string> correctChoiceArray = new List<string>();
        List<string> correctAnswerArray = new List<string>();
        int questionType = 0;
        bool includesImage = false;

        public ItemForm(string id)
        {
            InitializeComponent();
            identifier = id;
            LoadItemInfo();
        }

        public void LoadItemInfo()
        {
            IdentifierLabel.Text = "Identifikátor: " + identifier;
            questionType = GetQuestionType();

            FillPossibleAnswerLabel();
            FillCorrectAnswerLabel();

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        TitleLabel.Text = "Nadpis: " + xmlReader.GetAttribute("title");
                        LabelLabel.Text = "Označení: " + xmlReader.GetAttribute("label");
                    }
                }

                if (includesImage)
                {
                    if (xmlReader.Name == "div" && xmlReader.AttributeCount == 0 && xmlReader.NodeType != XmlNodeType.EndElement)//TODO 3: Předělat (?), div je potomkem prompt, viz TODO 2
                    {
                        QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                    }
                    if (xmlReader.Name == "img")
                    {
                        QuestionImage.ImageLocation = ("C:\\xampp\\exported\\items\\" + identifier + "\\" + xmlReader.GetAttribute("src"));
                    }
                }
                else
                {
                    if (xmlReader.Name == "prompt")
                    {
                        QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                    }
                }
                switch (questionType)//switch blok ve kterém čteme z XML, tzn jsme ve while cyklu
                {
                    case 0:
                        MessageBox.Show("Chyba: nepodporovaný nebo neznámý typ otázky.", "Nepodporovaná otázka", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        break;

                    case 1:
                        break;

                    case 2:
                        break;

                    case 3:
                        break;

                    case 4:
                        break;
                }
            }

            switch(questionType)//další switch blok ve kterém již nečteme z XML, tzn nejsme ve while cyklu
            {
                case 3:
                    break;

                case 4:

                    break;
            }
        }

        public int GetQuestionType()
        {
            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    if (xmlReader.GetAttribute("cardinality") == "ordered" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        questionType = 1;//Typ otázky = seřazení pojmů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        questionType = 2;//Typ otázky = více odpovědí (abc); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "pair")
                    {
                        questionType = 3;//Typ otázky = spojování párů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                    {
                        questionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                    {
                        questionType = 5;//Typ otázky = volná odpověď
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        questionType = 6;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
                    }
                }

                if (xmlReader.Name == "img")
                {
                    includesImage = true;
                }
            }
            return questionType;
        }

        public void FillPossibleAnswerLabel()
        {
            string possibleAnswers = "";
            string subquestions = "";
            int simpleMatchSetCounter = 0;

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                if(questionType == 4)
                {
                    if (xmlReader.Name == "simpleMatchSet")
                    {
                        simpleMatchSetCounter++;
                    }

                    if (xmlReader.Name == "simpleAssociableChoice")
                    {
                        if (simpleMatchSetCounter == 1)
                        {
                            string answerText = xmlReader.ReadElementContentAsString();
                            possibleAnswerArray.Add(answerText);
                        }
                        else if (simpleMatchSetCounter == 3)
                        {
                            string answerText = xmlReader.ReadElementContentAsString();
                            subquestionArray.Add(answerText);
                        }
                    }
                }
                else
                {
                    if (xmlReader.Name == "simpleChoice" || xmlReader.Name == "simpleAssociableChoice")
                    {
                        string answerText = xmlReader.ReadElementContentAsString();
                        possibleAnswerArray.Add(answerText);
                    }
                }
            }

            if (questionType == 4)
            {
                foreach (string answer in possibleAnswerArray)
                {
                    possibleAnswers += answer + "\n";
                }
                PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;

                int subquestionArrayIterator = 0;
                foreach (string answer in subquestionArray)
                {
                    if (subquestionArrayIterator != subquestionArray.Count - 1)
                    {
                        subquestions += answer + ", ";
                    }
                    else
                    {
                        subquestions += answer;
                    }
                    subquestionArrayIterator++;
                }
                //TODO 5: Šlo by to řešit i přes jeden label, tzn. přidáme text do QuestionLabelu, ale myslím si že tohle řešení ničemu nevadí
                SubquestionLabel.Visible = true;
                SubquestionLabel.Text = "(" + subquestions + ")";
            }
            else if(questionType == 5)
            {
                PossibleAnswerLabel.Text = "Jedná se o otevřenou otázku, neobsahuje výběr z možností, odpovědi je nutné ověřit manuálně.\n" + possibleAnswers;
            }
            else
            {
                foreach (string answer in possibleAnswerArray)
                {
                    possibleAnswers += answer + "\n";
                }
                PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;
            }
        }

        public void FillCorrectAnswerLabel()
        {
            string correctAnswer = "";
            List<string> identifierToString = new List<string>();
            identifierToString.Add(" ");
            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                if(questionType == 3)
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 7: viz TODO 2
                        {
                            string[] orderedCorrectChoices = value.Split(' ');
                            correctChoiceArray.Add(orderedCorrectChoices[0]);
                            correctChoiceArray.Add(orderedCorrectChoices[1]);
                            correctAnswerArray.Add(" ");//placeholder
                            correctAnswerArray.Add(" ");//placeholder
                        }
                    }

                    if (xmlReader.Name == "simpleAssociableChoice")
                    {
                        int i = 0;
                        foreach (string answer in correctChoiceArray)
                        {
                            if (xmlReader.GetAttribute("identifier") == answer)
                            {
                                string answerText = xmlReader.ReadElementContentAsString();
                                correctAnswerArray[i] = answerText;
                            }
                            i++;
                        }
                    }
                }
                else if (questionType == 4)
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 8: viz TODO 2
                        {
                            string[] orderedCorrectChoices = value.Split(' ');
                            correctChoiceArray.Add(orderedCorrectChoices[1]);
                            correctChoiceArray.Add(orderedCorrectChoices[0]);
                        }
                    }

                    if (xmlReader.Name == "simpleAssociableChoice")
                    {
                        string answerText = xmlReader.ReadElementContentAsString();
                        identifierToString.Add(answerText);
                    }
                }
                else if(questionType == 5)
                {
                    CorrectAnswerLabel.Visible = false;
                }
                else
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 2: Později předělat kód, ať to zjišťuje jestli jsme v correctResponse (správně) nebo v defaultValue (špatně)
                        {
                            correctChoiceArray.Add(value);
                            correctAnswerArray.Add(" ");//placeholder
                        }
                    }

                    if (xmlReader.Name == "simpleChoice")
                    {
                        int i = 0;
                        foreach (string answer in correctChoiceArray)
                        {
                            if (xmlReader.GetAttribute("identifier") == answer)
                            {
                                string answerText = xmlReader.ReadElementContentAsString();
                                correctAnswerArray[i] = answerText;
                            }
                            i++;
                        }
                    }
                }
            }

            //TODO 6: Zde máme sudý počet pojmů, a máme z nich vytvořit dvojice.
            //Já využívám té sudosti a mám všechny možnosti v poli stringů, možná by bylo vhodnejší použití pole tuplů stringů
            if (questionType == 3)
            {
                int answerNumber = 0;
                foreach (string answer in correctAnswerArray)
                {
                    if (answerNumber % 2 == 1)
                    {
                        correctAnswer += answer + "\n";
                    }
                    else
                    {
                        correctAnswer += answer + " -> ";
                    }
                    answerNumber++;
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
            else if(questionType == 4)
            {
                foreach(string answer in correctChoiceArray)
                {
                    string[] splitChoiceByUnderscore = answer.Split('_');
                    correctAnswerArray.Add(identifierToString[int.Parse(splitChoiceByUnderscore[1])]);
                }
                int answerNumber = 0;
                foreach (string answer in correctAnswerArray)
                {
                    if (answerNumber % 2 == 1)
                    {
                        correctAnswer += answer + "\n";
                    }
                    else
                    {
                        correctAnswer += answer + " -> ";
                    }
                    answerNumber++;
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
            else
            {
                foreach (string answer in correctAnswerArray)
                {
                    correctAnswer += answer + "\n";
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new ItemsForm().Show();
            Hide();
        }
    }
}
