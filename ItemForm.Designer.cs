
namespace TAO_Enhancer
{
    partial class ItemForm
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
            this.NumberIdentifierLabel = new System.Windows.Forms.Label();
            this.TitleLabel = new System.Windows.Forms.Label();
            this.LabelLabel = new System.Windows.Forms.Label();
            this.QuestionLabel = new System.Windows.Forms.Label();
            this.CorrectAnswerLabel = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.QuestionImage = new System.Windows.Forms.PictureBox();
            this.PossibleAnswerLabel = new System.Windows.Forms.Label();
            this.SubitemCB = new System.Windows.Forms.ComboBox();
            this.ResponseIdentifierLabel = new System.Windows.Forms.Label();
            this.ItemGB = new System.Windows.Forms.GroupBox();
            this.NameIdentifierLabel = new System.Windows.Forms.Label();
            this.AmountOfSubquestionsLabel = new System.Windows.Forms.Label();
            this.SubitemGB = new System.Windows.Forms.GroupBox();
            this.QuestionTypeLabel = new System.Windows.Forms.Label();
            this.SubitemLabel = new System.Windows.Forms.Label();
            this.StudentsAnswerGB = new System.Windows.Forms.GroupBox();
            this.StudentsAnswerLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.QuestionImage)).BeginInit();
            this.ItemGB.SuspendLayout();
            this.SubitemGB.SuspendLayout();
            this.StudentsAnswerGB.SuspendLayout();
            this.SuspendLayout();
            // 
            // NumberIdentifierLabel
            // 
            this.NumberIdentifierLabel.AutoSize = true;
            this.NumberIdentifierLabel.Location = new System.Drawing.Point(16, 37);
            this.NumberIdentifierLabel.Name = "NumberIdentifierLabel";
            this.NumberIdentifierLabel.Size = new System.Drawing.Size(126, 15);
            this.NumberIdentifierLabel.TabIndex = 0;
            this.NumberIdentifierLabel.Text = "NumberIdentifierLabel";
            // 
            // TitleLabel
            // 
            this.TitleLabel.AutoSize = true;
            this.TitleLabel.Location = new System.Drawing.Point(16, 98);
            this.TitleLabel.Name = "TitleLabel";
            this.TitleLabel.Size = new System.Drawing.Size(57, 15);
            this.TitleLabel.TabIndex = 1;
            this.TitleLabel.Text = "TitleLabel";
            // 
            // LabelLabel
            // 
            this.LabelLabel.AutoSize = true;
            this.LabelLabel.Location = new System.Drawing.Point(16, 133);
            this.LabelLabel.Name = "LabelLabel";
            this.LabelLabel.Size = new System.Drawing.Size(63, 15);
            this.LabelLabel.TabIndex = 2;
            this.LabelLabel.Text = "LabelLabel";
            // 
            // QuestionLabel
            // 
            this.QuestionLabel.AutoSize = true;
            this.QuestionLabel.Location = new System.Drawing.Point(15, 402);
            this.QuestionLabel.Name = "QuestionLabel";
            this.QuestionLabel.Size = new System.Drawing.Size(83, 15);
            this.QuestionLabel.TabIndex = 3;
            this.QuestionLabel.Text = "QuestionLabel";
            // 
            // CorrectAnswerLabel
            // 
            this.CorrectAnswerLabel.AutoSize = true;
            this.CorrectAnswerLabel.Location = new System.Drawing.Point(297, 451);
            this.CorrectAnswerLabel.Name = "CorrectAnswerLabel";
            this.CorrectAnswerLabel.Size = new System.Drawing.Size(113, 15);
            this.CorrectAnswerLabel.TabIndex = 4;
            this.CorrectAnswerLabel.Text = "CorrectAnswerLabel";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(458, 631);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 5;
            this.button1.Text = "Zpět";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // QuestionImage
            // 
            this.QuestionImage.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.QuestionImage.Location = new System.Drawing.Point(15, 176);
            this.QuestionImage.Name = "QuestionImage";
            this.QuestionImage.Size = new System.Drawing.Size(395, 198);
            this.QuestionImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.QuestionImage.TabIndex = 6;
            this.QuestionImage.TabStop = false;
            // 
            // PossibleAnswerLabel
            // 
            this.PossibleAnswerLabel.AutoSize = true;
            this.PossibleAnswerLabel.Location = new System.Drawing.Point(15, 451);
            this.PossibleAnswerLabel.Name = "PossibleAnswerLabel";
            this.PossibleAnswerLabel.Size = new System.Drawing.Size(117, 15);
            this.PossibleAnswerLabel.TabIndex = 7;
            this.PossibleAnswerLabel.Text = "PossibleAnswerLabel";
            // 
            // SubitemCB
            // 
            this.SubitemCB.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.SubitemCB.Enabled = false;
            this.SubitemCB.FormattingEnabled = true;
            this.SubitemCB.Location = new System.Drawing.Point(15, 67);
            this.SubitemCB.Name = "SubitemCB";
            this.SubitemCB.Size = new System.Drawing.Size(395, 23);
            this.SubitemCB.TabIndex = 9;
            this.SubitemCB.SelectedIndexChanged += new System.EventHandler(this.SubitemCB_SelectedIndexChanged);
            // 
            // ResponseIdentifierLabel
            // 
            this.ResponseIdentifierLabel.AutoSize = true;
            this.ResponseIdentifierLabel.Location = new System.Drawing.Point(15, 141);
            this.ResponseIdentifierLabel.Name = "ResponseIdentifierLabel";
            this.ResponseIdentifierLabel.Size = new System.Drawing.Size(132, 15);
            this.ResponseIdentifierLabel.TabIndex = 10;
            this.ResponseIdentifierLabel.Text = "ResponseIdentifierLabel";
            // 
            // ItemGB
            // 
            this.ItemGB.Controls.Add(this.NameIdentifierLabel);
            this.ItemGB.Controls.Add(this.AmountOfSubquestionsLabel);
            this.ItemGB.Controls.Add(this.NumberIdentifierLabel);
            this.ItemGB.Controls.Add(this.TitleLabel);
            this.ItemGB.Controls.Add(this.LabelLabel);
            this.ItemGB.Location = new System.Drawing.Point(12, 30);
            this.ItemGB.Name = "ItemGB";
            this.ItemGB.Size = new System.Drawing.Size(279, 269);
            this.ItemGB.TabIndex = 11;
            this.ItemGB.TabStop = false;
            this.ItemGB.Text = "Otázka";
            // 
            // NameIdentifierLabel
            // 
            this.NameIdentifierLabel.AutoSize = true;
            this.NameIdentifierLabel.Location = new System.Drawing.Point(16, 67);
            this.NameIdentifierLabel.Name = "NameIdentifierLabel";
            this.NameIdentifierLabel.Size = new System.Drawing.Size(114, 15);
            this.NameIdentifierLabel.TabIndex = 4;
            this.NameIdentifierLabel.Text = "NameIdentifierLabel";
            // 
            // AmountOfSubquestionsLabel
            // 
            this.AmountOfSubquestionsLabel.AutoSize = true;
            this.AmountOfSubquestionsLabel.Location = new System.Drawing.Point(16, 168);
            this.AmountOfSubquestionsLabel.Name = "AmountOfSubquestionsLabel";
            this.AmountOfSubquestionsLabel.Size = new System.Drawing.Size(163, 15);
            this.AmountOfSubquestionsLabel.TabIndex = 3;
            this.AmountOfSubquestionsLabel.Text = "AmountOfSubquestionsLabel";
            // 
            // SubitemGB
            // 
            this.SubitemGB.Controls.Add(this.QuestionTypeLabel);
            this.SubitemGB.Controls.Add(this.SubitemLabel);
            this.SubitemGB.Controls.Add(this.SubitemCB);
            this.SubitemGB.Controls.Add(this.ResponseIdentifierLabel);
            this.SubitemGB.Controls.Add(this.PossibleAnswerLabel);
            this.SubitemGB.Controls.Add(this.QuestionImage);
            this.SubitemGB.Controls.Add(this.QuestionLabel);
            this.SubitemGB.Controls.Add(this.CorrectAnswerLabel);
            this.SubitemGB.Location = new System.Drawing.Point(322, 30);
            this.SubitemGB.Name = "SubitemGB";
            this.SubitemGB.Size = new System.Drawing.Size(625, 562);
            this.SubitemGB.TabIndex = 12;
            this.SubitemGB.TabStop = false;
            this.SubitemGB.Text = "Podotázka";
            // 
            // QuestionTypeLabel
            // 
            this.QuestionTypeLabel.AutoSize = true;
            this.QuestionTypeLabel.Location = new System.Drawing.Point(15, 110);
            this.QuestionTypeLabel.Name = "QuestionTypeLabel";
            this.QuestionTypeLabel.Size = new System.Drawing.Size(107, 15);
            this.QuestionTypeLabel.TabIndex = 12;
            this.QuestionTypeLabel.Text = "QuestionTypeLabel";
            // 
            // SubitemLabel
            // 
            this.SubitemLabel.AutoSize = true;
            this.SubitemLabel.Location = new System.Drawing.Point(15, 37);
            this.SubitemLabel.Name = "SubitemLabel";
            this.SubitemLabel.Size = new System.Drawing.Size(79, 15);
            this.SubitemLabel.TabIndex = 11;
            this.SubitemLabel.Text = "SubitemLabel";
            // 
            // StudentsAnswerGB
            // 
            this.StudentsAnswerGB.Controls.Add(this.StudentsAnswerLabel);
            this.StudentsAnswerGB.Location = new System.Drawing.Point(973, 30);
            this.StudentsAnswerGB.Name = "StudentsAnswerGB";
            this.StudentsAnswerGB.Size = new System.Drawing.Size(406, 181);
            this.StudentsAnswerGB.TabIndex = 13;
            this.StudentsAnswerGB.TabStop = false;
            this.StudentsAnswerGB.Text = "Vaše odpověď";
            this.StudentsAnswerGB.Visible = false;
            // 
            // StudentsAnswerLabel
            // 
            this.StudentsAnswerLabel.AutoSize = true;
            this.StudentsAnswerLabel.Location = new System.Drawing.Point(7, 37);
            this.StudentsAnswerLabel.Name = "StudentsAnswerLabel";
            this.StudentsAnswerLabel.Size = new System.Drawing.Size(120, 15);
            this.StudentsAnswerLabel.TabIndex = 0;
            this.StudentsAnswerLabel.Text = "StudentsAnswerLabel";
            // 
            // ItemForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1413, 728);
            this.Controls.Add(this.StudentsAnswerGB);
            this.Controls.Add(this.SubitemGB);
            this.Controls.Add(this.ItemGB);
            this.Controls.Add(this.button1);
            this.Name = "ItemForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "ItemForm";
            ((System.ComponentModel.ISupportInitialize)(this.QuestionImage)).EndInit();
            this.ItemGB.ResumeLayout(false);
            this.ItemGB.PerformLayout();
            this.SubitemGB.ResumeLayout(false);
            this.SubitemGB.PerformLayout();
            this.StudentsAnswerGB.ResumeLayout(false);
            this.StudentsAnswerGB.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label NumberIdentifierLabel;
        private System.Windows.Forms.Label TitleLabel;
        private System.Windows.Forms.Label LabelLabel;
        private System.Windows.Forms.Label QuestionLabel;
        private System.Windows.Forms.Label CorrectAnswerLabel;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PictureBox QuestionImage;
        private System.Windows.Forms.Label PossibleAnswerLabel;
        private System.Windows.Forms.ComboBox SubitemCB;
        private System.Windows.Forms.Label ResponseIdentifierLabel;
        private System.Windows.Forms.GroupBox ItemGB;
        private System.Windows.Forms.GroupBox SubitemGB;
        private System.Windows.Forms.Label SubitemLabel;
        private System.Windows.Forms.Label QuestionTypeLabel;
        private System.Windows.Forms.Label AmountOfSubquestionsLabel;
        private System.Windows.Forms.GroupBox StudentsAnswerGB;
        private System.Windows.Forms.Label NameIdentifierLabel;
        private System.Windows.Forms.Label StudentsAnswerLabel;
    }
}