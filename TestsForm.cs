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
    public partial class TestsForm : Form
    {
        int chosenTest = -1;
        List<(string, string)> itemIdentifiers = new List<(string, string)>();

        public TestsForm()
        {
            InitializeComponent();
            LoadTests();
        }

        public void LoadTests()
        {
            int gridViewRow = 0;
            string subDirectory = "";

            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\tests"))
            {
                string nameIdentifier = Path.GetFileName(directory);
                string title = "";
                string numberidentifier = "";

                foreach(var directory_ in Directory.GetDirectories(directory + "\\tests"))
                {
                    numberidentifier = Path.GetFileName(directory_);
                    subDirectory = directory_;
                }
                itemIdentifiers.Add((nameIdentifier, numberidentifier));

                XmlReader xmlReader = XmlReader.Create(subDirectory + "\\test.xml");
                while (xmlReader.Read())
                {
                    if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                    {
                        if (xmlReader.HasAttributes)
                        {
                            title = xmlReader.GetAttribute("title");
                            numberidentifier = xmlReader.GetAttribute("identifier");
                        }
                    }
                }

                TestsGridView.Rows.Add();
                TestsGridView.Rows[gridViewRow].Cells[0].Value = nameIdentifier;
                TestsGridView.Rows[gridViewRow].Cells[1].Value = numberidentifier;
                TestsGridView.Rows[gridViewRow].Cells[2].Value = title;
                gridViewRow++;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (chosenTest == -1)
            {
                MessageBox.Show("Chyba - nevybral jste žadnou otázku. Prosím vyberte otázku kliknutím na příslušný řádek.", "Nebyla vybrána otázka", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                new TestForm(itemIdentifiers[chosenTest], true, "", "").Show();
                Hide();
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new TeacherForm().Show();
            Hide();
        }

        private void TestsGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            chosenTest = TestsGridView.CurrentCell.RowIndex;
        }
    }
}
