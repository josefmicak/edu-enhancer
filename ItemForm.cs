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
                switch (questionType)
                {
                    case 0:
                        MessageBox.Show("Chyba: nepodporovaný nebo neznámý typ otázky.", "Nepodporovaná otázka", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        break;

                    case 1:
                        break;

                    case 2:
                        break;
                }
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
            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "simpleChoice")
                {
                    string answerText = xmlReader.ReadElementContentAsString();
                    possibleAnswerArray.Add(answerText);
                }
            }

            foreach (string answer in possibleAnswerArray)
            {
                possibleAnswers += answer + "\n";
            }
            PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;
        }

        public void FillCorrectAnswerLabel()
        {
            string correctAnswer = "";
            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
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

                //udělám nové pole o tolika prvcích kolik je správných odpovědí, a postupně dosadím ty odpovědi na příslušná místa
                if (xmlReader.Name == "simpleChoice")
                 {
                    int i = 0;
                    foreach (string answer in correctChoiceArray)
                     {
                         Debug.WriteLine(xmlReader.GetAttribute("identifier") + " == " + answer);
                         if (xmlReader.GetAttribute("identifier") == answer)
                         {
                             string answerText = xmlReader.ReadElementContentAsString();
                             correctAnswerArray[i] = answerText;
                         }
                         i++;
                     }
                 }
            }

            foreach (string answer in correctAnswerArray)
            {
                correctAnswer += answer + "\n";
            }
            Debug.WriteLine("AA: " + correctAnswerArray.Count);
            CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new ItemsForm().Show();
            Hide();
        }
    }
}
