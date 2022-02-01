
namespace TAO_Enhancer
{
    partial class TeacherForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button2 = new System.Windows.Forms.Button();
            this.ManageTestTemplates = new System.Windows.Forms.Button();
            this.ManageTestResults = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(362, 408);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 1;
            this.button2.Text = "Zpět";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // ManageTestTemplates
            // 
            this.ManageTestTemplates.Location = new System.Drawing.Point(85, 106);
            this.ManageTestTemplates.Name = "ManageTestTemplates";
            this.ManageTestTemplates.Size = new System.Drawing.Size(167, 23);
            this.ManageTestTemplates.TabIndex = 2;
            this.ManageTestTemplates.Text = "Správa zadání testů";
            this.ManageTestTemplates.UseVisualStyleBackColor = true;
            this.ManageTestTemplates.Click += new System.EventHandler(this.button3_Click);
            // 
            // ManageTestResults
            // 
            this.ManageTestResults.Location = new System.Drawing.Point(85, 181);
            this.ManageTestResults.Name = "ManageTestResults";
            this.ManageTestResults.Size = new System.Drawing.Size(167, 23);
            this.ManageTestResults.TabIndex = 3;
            this.ManageTestResults.Text = "Správa vyřešených testů";
            this.ManageTestResults.UseVisualStyleBackColor = true;
            this.ManageTestResults.Click += new System.EventHandler(this.ManageTestResults_Click);
            // 
            // TeacherForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.ManageTestResults);
            this.Controls.Add(this.ManageTestTemplates);
            this.Controls.Add(this.button2);
            this.Name = "TeacherForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "TeacherForm";
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button ManageTestTemplates;
        private System.Windows.Forms.Button ManageTestResults;
    }
}