using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;

namespace TAO_Enhancer
{
    public partial class TestForm : Form
    {
        string testNameIdentifier = "";
        string testNumberIdentifier = "";
        int chosenItem = -1;
        string itemNameIdentifier = "";
        string itemNumberIdentifier = "";
        bool isTeacher = true;
        string deliveryExecutionIdentifier = "";
        string studentIdentifier = "";

        public TestForm((string, string) id, bool requestOrigin, string attemptIdentifier, string studentID)
        {
            InitializeComponent();
            testNameIdentifier = id.Item1;
            testNumberIdentifier = id.Item2;
            if(!requestOrigin)
            {
                isTeacher = false;
                deliveryExecutionIdentifier = attemptIdentifier;
                studentIdentifier = studentID;
            }
            LoadTestInfo();
        }

        public void LoadTestInfo()
        {
            TestFolderNameLabel.Text = "Jméno složky: " + testNameIdentifier;
            TestNumberIdentifierLabel.Text = "Číselný identifikátor testu: " + testNumberIdentifier;

            string testitemIdentifier;
            string itemIdentifier;
            string testPart = "";
            string testSection = "";
            int amountOfItems = 0;

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier + "\\test.xml");
            while (xmlReader.Read())
            {
                if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                {
                    if (xmlReader.HasAttributes)
                    {
                        TestTitleLabel.Text = "Nadpis testu: " + xmlReader.GetAttribute("title");
                        TestNameIdentifierLabel.Text = "Jmenný identifikátor testu: " + xmlReader.GetAttribute("identifier");
                    }
                }

                if(xmlReader.Name == "testPart")
                {
                    testPart = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentSection")
                {
                    testSection = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testitemIdentifier = xmlReader.GetAttribute("identifier");
                    itemIdentifier = xmlReader.GetAttribute("href");
                    string[] itemIdentifierSplit = itemIdentifier.Split("/");
                    ItemsGridView.Rows.Add();
                    ItemsGridView.Rows[amountOfItems].Cells[0].Value = testPart;
                    ItemsGridView.Rows[amountOfItems].Cells[1].Value = testSection;
                    ItemsGridView.Rows[amountOfItems].Cells[2].Value = testitemIdentifier;
                    ItemsGridView.Rows[amountOfItems].Cells[3].Value = itemIdentifierSplit[3];
                    amountOfItems++;
                }
            }
            AmountOfQuestionsLabel.Text = "Počet otázek: " + amountOfItems;

            int testPoints = 0;
            bool testPointsDetermined = true;

            for(int i = 0; i < ItemsGridView.Rows.Count; i++)
            {
                bool questionPointsDetermined = false;
                int questionPoints = 0;
                string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + ItemsGridView.Rows[i].Cells[3].Value;
             
                foreach (var file in Directory.GetFiles(itemParentPath))
                {
                    if (Path.GetFileName(file) == "Points.txt")
                    {
                        questionPointsDetermined = true;
                    }
                }

                if(questionPointsDetermined)
                {
                    string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
                    for (int j = 0; j < importedFileLines.Length; j++)
                    {
                        string[] splitImportedFileLineBySemicolon = importedFileLines[j].Split(";");

                        if (splitImportedFileLineBySemicolon[1] == "N/A")
                        {
                            questionPointsDetermined = false;
                        }
                        else
                        {
                            questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                        }
                    }
                }

                if(!questionPointsDetermined)
                {
                    testPointsDetermined = false;
                    ItemsGridView.Rows[i].Cells[4].Value = "N/A";
                }
                else
                {
                    testPoints += questionPoints;
                    ItemsGridView.Rows[i].Cells[4].Value = questionPoints.ToString();
                }
            }

            if(!testPointsDetermined)
            {
                TestPointsLabel.Text = "Počet bodů za test: N/A";
            }
            else
            {
                TestPointsLabel.Text = "Počet bodů za test: " + testPoints.ToString();
            }
        }

        public void LoadItemInfo()
        {
            ItemNameIdentifierLabel.Text = "Jmenný identifikátor otázky: " + ItemsGridView.Rows[chosenItem].Cells[2].Value.ToString();
            ItemNumberIdentifierLabel.Text = "Číselný identifikátor otázky: " + ItemsGridView.Rows[chosenItem].Cells[3].Value.ToString();
            itemNameIdentifier = ItemsGridView.Rows[chosenItem].Cells[2].Value.ToString();
            itemNumberIdentifier = ItemsGridView.Rows[chosenItem].Cells[3].Value.ToString();
            ItemPointsLabel.Text = "Počet bodů za otázku: " + ItemsGridView.Rows[chosenItem].Cells[4].Value.ToString();

            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        ItemTitleLabel.Text = "Nadpis otázky: " + xmlReader.GetAttribute("title");
                        ItemLabelLabel.Text = "Označení otázky: " + xmlReader.GetAttribute("label");
                    }
                }
            }
        }

        public string GetItemPath()
        {
            return "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\qti.xml";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(isTeacher)
            {
                new TestsForm().Show();
                Hide();
            }
            else
            {
                new ResultForm(studentIdentifier).Show();
                Hide();
            }
        }

        private void ItemsGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            chosenItem = ItemsGridView.CurrentCell.RowIndex;
            LoadItemInfo();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new ItemForm(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier, testNumberIdentifier, isTeacher, deliveryExecutionIdentifier, studentIdentifier).Show();
            Hide();
        }
    }
}
